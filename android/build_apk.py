#!/usr/bin/env python3
"""Build an installable wrapper; keep the reusable signing key outside Git."""
import argparse
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
from urllib.parse import urlsplit

PROJECT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='https://amicbeam.fun/card-game', help='完整桌游网址，默认使用线上大厅')
    parser.add_argument('--debug', action='store_true', help='构建调试包，允许 HTTP 本地联调')
    args = parser.parse_args()
    entry = urlsplit(args.url)
    if (entry.scheme not in ('http', 'https') or not entry.hostname
            or entry.username is not None or entry.password is not None):
        parser.error('--url 必须是完整 HTTP(S) 地址，且不能包含账号密码')
    if not args.debug and entry.scheme != 'https':
        parser.error('正式安装包必须使用 HTTPS 地址')

    env = os.environ.copy()
    studio_jdk = Path('/Applications/Android Studio.app/Contents/jbr/Contents/Home')
    if not env.get('JAVA_HOME') and studio_jdk.is_dir():
        env['JAVA_HOME'] = str(studio_jdk)
    sdk = Path.home() / 'Library/Android/sdk'
    if not env.get('ANDROID_HOME') and sdk.is_dir():
        env['ANDROID_HOME'] = str(sdk)

    if not args.debug:
        properties_path = env.get('NTE_ANDROID_SIGNING_PROPERTIES')
        if not properties_path:
            key_dir = Path.home() / '.local/share/nte-esper-duel-android'
            key_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            properties = key_dir / 'signing.properties'
            keystore = key_dir / 'release.p12'
            if not properties.exists():
                if keystore.exists():
                    sys.exit('签名密码文件缺失，请恢复备份；已有签名密钥不会被覆盖。')
                password = secrets.token_hex(24)
                with properties.open('x', encoding='utf-8') as output:
                    os.chmod(properties, 0o600)
                    output.write(f'storeFile={keystore.as_posix()}\nstorePassword={password}\n'
                                 f'keyAlias=esper-duel\nkeyPassword={password}\n')
            if not keystore.exists():
                values = dict(line.strip().split('=', 1) for line in properties.read_text().splitlines() if '=' in line)
                key_env = env.copy()
                key_env['NTE_ANDROID_KEY_PASSWORD'] = values['storePassword']
                keytool = (str(Path(env['JAVA_HOME']) / 'bin/keytool')
                           if env.get('JAVA_HOME') else 'keytool')
                subprocess.run([
                    keytool, '-genkeypair', '-noprompt', '-storetype', 'PKCS12',
                    '-keystore', str(keystore), '-alias', 'esper-duel',
                    '-keyalg', 'RSA', '-keysize', '3072', '-validity', '10000',
                    '-dname', 'CN=Esper Duel',
                    '-storepass:env', 'NTE_ANDROID_KEY_PASSWORD',
                    '-keypass:env', 'NTE_ANDROID_KEY_PASSWORD',
                ], env=key_env, check=True)
                os.chmod(keystore, 0o600)
            properties_path = str(properties)
        env['NTE_ANDROID_SIGNING_PROPERTIES'] = properties_path

    variant = 'debug' if args.debug else 'release'
    wrapper = 'gradlew.bat' if os.name == 'nt' else './gradlew'
    subprocess.run([wrapper, '--no-daemon', f'-PgameUrl={args.url}',
                    f':app:assemble{variant.title()}', f':app:lint{variant.title()}'],
                   cwd=PROJECT, env=env, check=True)
    source = PROJECT / f'app/build/outputs/apk/{variant}/app-{variant}.apk'
    destination = PROJECT.parent / f'dist/android/esper-duel-{variant}.apk'
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(f'APK: {destination}')


if __name__ == '__main__':
    main()
