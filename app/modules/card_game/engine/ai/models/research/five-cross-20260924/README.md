# 五套交叉训练的未验收研究权重

本目录保存仓库原本缺少的真红（`zhenhong`）与浊燃（`murk`）数值文件，来源为 `artifacts/rl-evals/five-cross-20260924/formal/generations/cycle-0011/`，原始 JSON 清单及 `.npz` SHA 保持不变。两套权重**未通过质量验收**：`automatic_serving_approval=false`，旧运行规则指纹与当前源码不同，2026-09-25 的质量决定为 `NO_GO_FOR_EFFECTIVE_FORMAL_TRAINING`。用户随后明确授权把仓库旧策略没有的构筑开放为实验性高级人机；`experimental_cross_model.py` 只对本目录两份固定 SHA、固定新旧规则指纹和相同观察／动作列提供显式实验加载。不要复制到服务根目录或改写原清单。

真红 SHA-256：`3c1258b60d9e985a30da55b6d98bc6526064c450a0292ece64c378ebc7f428e1`；浊燃 SHA-256：`744b4b9d71cca135df6f84ecb0479979cfc1ae6e922e8f62a535a97f902d9f51`。实验性网页对战授权与训练质量验收分开；模型不会因此成为正式推荐策略。
