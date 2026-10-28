# 排轴增伤分类核对

三类是同一个增伤乘区内部的展示拆分，仍按 `1＋通用增伤＋动作增伤＋其他增伤＋本次属性增伤` 相乘；不把三类彼此相乘。

| 项目 | 旧统计 | 三类设计 |
| --- | --- | --- |
| 全伤、通伤词条、增伤赐福及全动作增益 | 通用增伤 | 通用增伤 |
| 限定技能/动作的全伤配置及动作自身强化 | 通用增伤 | 动作增伤 |
| 普攻、闪反、变轨、终结、追击/追加攻击、附着 | 其他增伤 | 动作增伤 |
| 残虹狩 | 其他增伤 | 其他增伤 |
| 负面状态计数、残虹灭末段层蚀、灵可谐频与无法环合加成 | 通用增伤 | 其他增伤 |
| 光、灵、咒、暗、魂、相 | 属性增伤 | 属性增伤，按实际伤害属性选择 |
| 心灵 | 其他增伤 | 第7种属性增伤，保留存档键 mind_dmg |

属性是伤害属性，不改变角色所属六属性和环合组合。暴击、暴伤、最终增伤、基础倍率提升、减防、穿防、减抗、环合强度和倾陷专用增伤均不并入这三类。

「静候彩铃」按通用光属性减抗处理，角色伤害、周期伤害、对应属性的环合与倾陷均可受益；覆纹与浸染保留既有本体增幅算法。

以下为当前有效配置的非零条目及动态输出；精炼档位合并在同一行，数值使用内部小数比例。条件、对象和层数仍由各规则控制，不代表整轴全程生效。旧统计列表示拆分前的归属。历史 characters.modifiers 和纯说明文本不参与计算，未列入。

| 配置 | 条目 | 规则/记录ID | 字段 | 数值/动态规则 | 旧统计 | 三类设计 |
| --- | --- | --- | --- | --- | --- | --- |
| actions.json | 达芙蒂尔·1层e | `action_5206179376` | `action_dmg` | 0.5 | 通用增伤 | 动作增伤 |
| actions.json | 达芙蒂尔·2层e | `action_351bba97e4` | `action_dmg` | 1 | 通用增伤 | 动作增伤 |
| arc_refinements.json | 「倾世之雨」 | `arc_facc69ac01` | `skill_dmg` | 0.3 / 0.33 / 0.36 / 0.39 / 0.42 | 其他增伤 | 动作增伤 |
| arc_refinements.json | 「倾世之雨」 | `arc_facc69ac01` | `ultimate_dmg` | 0.3 / 0.33 / 0.36 / 0.39 / 0.42 | 其他增伤 | 动作增伤 |
| arc_refinements.json | 「我们。」 | `arc_7378181f20` | `basic_dmg` | 0.12 / 0.13 / 0.14 / 0.15 / 0.16 | 其他增伤 | 动作增伤 |
| arc_refinements.json | 「电音」狂欢 | `arc_b99b47cd11` | `skill_dmg` | 0.12 / 0.13 / 0.14 / 0.15 / 0.16 | 其他增伤 | 动作增伤 |
| arc_refinements.json | 「银河暂留」 | `arc_8fbb15cb26` | `element_dmg_魂` | 0.12 / 0.145 / 0.17 / 0.195 / 0.21 | 属性增伤 | 属性增伤 |
| arc_refinements.json | 不屈之绵 | `arc_0a681d8e81` | `all_dmg` | 0.25 / 0.3 / 0.35 / 0.4 / 0.45 | 通用增伤 | 通用增伤 |
| arc_refinements.json | 不屈之绵 | `arc_b22de80f07` | `all_dmg` | 0.25 / 0.3 / 0.35 / 0.4 / 0.45 | 通用增伤 | 通用增伤 |
| arc_refinements.json | 众人追寻之物 | `arc_d7e0691c5f` | `element_dmg_相` | 0.2 / 0.25 / 0.3 / 0.35 / 0.4 | 属性增伤 | 属性增伤 |
| arc_refinements.json | 光波眩晕，迷走狂欢：援护后增伤 | `arc_dazzle_support_all_dmg` | `all_dmg` | 0.18 / 0.21 / 0.24 / 0.27 / 0.3 | 通用增伤 | 通用增伤 |
| arc_refinements.json | 开始净空·纸翼战队 | `arc_paperplane_spirit_eq` | `action_dmg` | 0.2 / 0.2375 / 0.275 / 0.3125 / 0.35 | 通用增伤 | 动作增伤 |
| arc_refinements.json | 引爆全场：前台魂伤 | `arc_bit_game_foreground_psyche_dmg` | `element_dmg_魂` | 0.12 / 0.14 / 0.16 / 0.18 / 0.2 | 属性增伤 | 属性增伤 |
| arc_refinements.json | 引爆全场：普攻魂伤叠层 | `arc_bit_game_basic_psyche_stack` | `element_dmg_魂` | 0.02 / 0.023 / 0.026 / 0.029 / 0.032 | 属性增伤 | 属性增伤 |
| arc_refinements.json | 当心头顶：延滞下相伤 | `arc_overhead_delay_element` | `element_dmg_相` | 0.1 / 0.12 / 0.14 / 0.16 / 0.18 | 属性增伤 | 属性增伤 |
| arc_refinements.json | 思考喵·满精：方斯叠满光伤 | `arc_thinking_cat_full_fons_light` | `element_dmg_光` | 0.25 / 0.3 / 0.35 / 0.4 / 0.45 | 属性增伤 | 属性增伤 |
| arc_refinements.json | 思考喵：方斯叠满光伤 | `arc_thinking_cat_fons_light` | `element_dmg_光` | 0.25 / 0.3 / 0.35 / 0.4 / 0.45 | 属性增伤 | 属性增伤 |
| arc_refinements.json | 月的对跖点·变轨增伤 | `arc_akane_skill` | `skill_dmg` | 0.28 / 0.35 / 0.42 / 0.49 / 0.56 | 其他增伤 | 动作增伤 |
| arc_refinements.json | 月的对跖点·普攻增伤 | `arc_akane_basic` | `basic_dmg` | 0.28 / 0.35 / 0.42 / 0.49 / 0.56 | 其他增伤 | 动作增伤 |
| arc_refinements.json | 月的对跖点·相伤 | `arc_akane_element` | `element_dmg_相` | 0.2 / 0.25 / 0.3 / 0.35 / 0.4 | 属性增伤 | 属性增伤 |
| arc_refinements.json | 欧拉欧拉！：普伤叠层 | `arc_ora_basic_stack` | `basic_dmg` | 0.02 / 0.024 / 0.028 / 0.032 / 0.036 | 其他增伤 | 动作增伤 |
| arc_refinements.json | 永恒圆舞曲：Q 后潜意伤 | `arc_waltz_q_mind` | `mind_dmg` | 0.1 / 0.12 / 0.14 / 0.16 / 0.18 | 其他增伤 | 属性增伤 |
| arc_refinements.json | 海特洛的安宁 | `arc_f5d1774a5a` | `all_dmg` | 0.15 / 0.175 / 0.2 / 0.225 / 0.25 | 通用增伤 | 通用增伤 |
| arc_refinements.json | 焰魂狂飙 | `arc_9758ceaf0a` | `element_dmg_相` | 0.15 / 0.175 / 0.2 / 0.225 / 0.25 | 属性增伤 | 属性增伤 |
| arc_refinements.json | 现实避难所 | `arc_f317782734` | `attach_dmg` | 0.1 / 0.1125 / 0.125 / 0.1375 / 0.15 | 其他增伤 | 动作增伤 |
| arc_refinements.json | 现实避难所 | `arc_f317782734` | `element_dmg_灵` | 0.15 / 0.175 / 0.2 / 0.225 / 0.25 | 属性增伤 | 属性增伤 |
| arc_refinements.json | 现实避难所：Q 后附着伤害 | `arc_reality_shelter_q_attach` | `attach_dmg` | 0.1 / 0.1125 / 0.125 / 0.1375 / 0.15 | 其他增伤 | 动作增伤 |
| arc_refinements.json | 穿过胭红蜃景满：Q 后光伤穿防 | `arc_crimson_mirage_full_q_light_def` | `element_dmg_光` | 0.32 / 0.368 / 0.416 / 0.464 / 0.512 | 属性增伤 | 属性增伤 |
| arc_refinements.json | 穿过胭红蜃景：Q 后光伤穿防 | `arc_crimson_mirage_q_light_def` | `element_dmg_光` | 0.32 / 0.368 / 0.416 / 0.464 / 0.512 | 属性增伤 | 属性增伤 |
| arc_refinements.json | 罪与罚 | `arc_heiyu_crime_punishment` | `element_dmg_魂` | 0.2 / 0.25 / 0.3 / 0.35 / 0.4 | 属性增伤 | 属性增伤 |
| arc_refinements.json | 错误的门：治疗后其他队友通用增伤 | `arc_wrong_door_heal_team_damage` | `all_dmg` | 0.15 / 0.1875 / 0.225 / 0.2625 / 0.3 | 通用增伤 | 通用增伤 |
| arc_refinements.json | 错误的门：治疗后灵伤 | `arc_wrong_door_heal_spirit` | `element_dmg_灵` | 0.3 / 0.375 / 0.45 / 0.525 / 0.6 | 属性增伤 | 属性增伤 |
| arc_refinements.json | 预备备：EQ 后普攻闪反增伤 | `arc_ready_eq_basic_dodge` | `basic_dmg` | 0.15 / 0.1875 / 0.225 / 0.2625 / 0.3 | 其他增伤 | 动作增伤 |
| arc_refinements.json | 预备备：EQ 后普攻闪反增伤 | `arc_ready_eq_basic_dodge` | `dodge_counter_dmg` | 0.15 / 0.1875 / 0.225 / 0.2625 / 0.3 | 其他增伤 | 动作增伤 |
| arc_refinements.json | 预备备：司令虎符对 Boss 增伤 | `arc_ready_tally_boss_damage` | `all_dmg` | 0.1 / 0.125 / 0.15 / 0.175 / 0.2 | 通用增伤 | 通用增伤 |
| arc_refinements.json | 鲸之歌 | `arc_d75aa15b91` | `all_dmg` | 0.04 / 0.0466666667 / 0.0533333333 / 0.06 / 0.0666666667 | 通用增伤 | 通用增伤 |
| arcs.json | 「倾世之雨」 | `arc_facc69ac01` | `skill_dmg` | 0.3 | 其他增伤 | 动作增伤 |
| arcs.json | 「倾世之雨」 | `arc_facc69ac01` | `ultimate_dmg` | 0.3 | 其他增伤 | 动作增伤 |
| arcs.json | 「我们。」 | `arc_7378181f20` | `basic_dmg` | 0.16 | 其他增伤 | 动作增伤 |
| arcs.json | 不屈之绵 | `arc_b22de80f07` | `all_dmg` | 0.25 | 通用增伤 | 通用增伤 |
| arcs.json | 不屈之绵·满精 | `arc_0a681d8e81` | `all_dmg` | 0.45 | 通用增伤 | 通用增伤 |
| arcs.json | 众人追寻之物 | `arc_d7e0691c5f` | `element_dmg_相` | 0.2 | 属性增伤 | 属性增伤 |
| arcs.json | 焰魂狂飙 | `arc_9758ceaf0a` | `element_dmg_相` | 0.15 | 属性增伤 | 属性增伤 |
| arcs.json | 现实避难所 | `arc_f317782734` | `attach_dmg` | 0.1 | 其他增伤 | 动作增伤 |
| arcs.json | 现实避难所 | `arc_f317782734` | `element_dmg_灵` | 0.15 | 属性增伤 | 属性增伤 |
| arcs.json | 电音狂欢 | `arc_b99b47cd11` | `skill_dmg` | 0.16 | 其他增伤 | 动作增伤 |
| arcs.json | 罪与罚 | `arc_heiyu_crime_punishment` | `element_dmg_魂` | 0.2 | 属性增伤 | 属性增伤 |
| arcs.json | 银河暂留 | `arc_8fbb15cb26` | `element_dmg_魂` | 0.12 | 属性增伤 | 属性增伤 |
| arcs.json | 鲸之歌 | `arc_d75aa15b91` | `all_dmg` | 0.04 | 通用增伤 | 通用增伤 |
| buffs.json | 11 您访问的地址已被永久转移 | `character_elyms_f` | `follow_dmg` | {"factor": 1, "source": "per_layer"} | 其他增伤 | 动作增伤 |
| buffs.json | 中继节点·盈蓄强化 | `character_elyms_energy_light` | `element_dmg_光` | 0.0375 | 属性增伤 | 属性增伤 |
| buffs.json | 伊洛伊满觉·退行队友增伤 | `character_iloy_full_regressed_teammate_damage` | `all_dmg` | 0.45 | 通用增伤 | 通用增伤 |
| buffs.json | 光波眩晕，迷走狂欢：援护后增伤 | `arc_dazzle_support_all_dmg` | `all_dmg` | 0.3 | 通用增伤 | 通用增伤 |
| buffs.json | 升腾之赤 | `character_zhenhong_ascendant_red` | `action_dmg` | 0.3 | 通用增伤 | 动作增伤 |
| buffs.json | 卡厄斯5觉·单目标追缉许可增伤 | `character_chaos_a5_license_damage` | `all_dmg` | 0.2 | 通用增伤 | 通用增伤 |
| buffs.json | 卡厄斯：追缉许可 | `character_chaos_pursuit_license` | `all_dmg` | 0.2 | 通用增伤 | 通用增伤 |
| buffs.json | 哈尼娅5觉·王牌魂伤 | `character_haniya_a5_ace_soul_damage` | `element_dmg_魂` | 0.3 | 属性增伤 | 属性增伤 |
| buffs.json | 哈索尔1觉·首次长按变轨闪送之力 | `character_hathor_a1_first_long_e_flash_force` | `all_dmg` | 0.01 | 通用增伤 | 通用增伤 |
| buffs.json | 哈索尔2觉·优评锁定闪送之力 | `character_hathor_a2_good_review_flash_force` | `all_dmg` | 0.01 | 通用增伤 | 通用增伤 |
| buffs.json | 哈索尔·闪送之力 | `character_hathor_flash_force` | `all_dmg` | 0.01 | 通用增伤 | 通用增伤 |
| buffs.json | 埃德嘉4觉·变轨强化 | `character_edgar_a4_e_damage` | `action_dmg` | 0.5 | 通用增伤 | 动作增伤 |
| buffs.json | 娜娜莉4觉·副手强化 | `character_nanali_offhand_damage` | `action_dmg` | 1.0 | 通用增伤 | 动作增伤 |
| buffs.json | 娜娜莉满觉·全伤 | `character_nanali_a6_damage` | `all_dmg` | 0.15 | 通用增伤 | 通用增伤 |
| buffs.json | 安魂曲2觉·援护入场增伤 | `character_requiem_a2_support_damage` | `all_dmg` | 0.15 | 通用增伤 | 通用增伤 |
| buffs.json | 小吱6觉·金谷强化 | `character_xiaozhi_a6_gold_damage` | `action_dmg` | 0.15 | 通用增伤 | 动作增伤 |
| buffs.json | 开始净空·纸翼战队 | `arc_paperplane_spirit_eq` | `action_dmg` | 0.35 | 通用增伤 | 动作增伤 |
| buffs.json | 引爆全场：前台魂伤 | `arc_bit_game_foreground_psyche_dmg` | `element_dmg_魂` | 0.12 | 属性增伤 | 属性增伤 |
| buffs.json | 引爆全场：普攻魂伤叠层 | `arc_bit_game_basic_psyche_stack` | `element_dmg_魂` | 0.02 | 属性增伤 | 属性增伤 |
| buffs.json | 当心头顶：延滞下相伤 | `arc_overhead_delay_element` | `element_dmg_相` | 0.18 | 属性增伤 | 属性增伤 |
| buffs.json | 思考喵·满精：方斯叠满光伤 | `arc_thinking_cat_full_fons_light` | `element_dmg_光` | 0.45 | 属性增伤 | 属性增伤 |
| buffs.json | 思考喵：方斯叠满光伤 | `arc_thinking_cat_fons_light` | `element_dmg_光` | 0.25 | 属性增伤 | 属性增伤 |
| buffs.json | 恶魔之血：魂反应增伤翻倍 | `cartridge_devils_blood_soul_reaction` | `all_dmg` | 0.18 | 通用增伤 | 通用增伤 |
| buffs.json | 早雾满觉·负面效果增伤 | `character_sagiri_a6_negative_damage` | `other_dmg` | {"damage_tags": ["DOT"], "effect_key": "other_dmg", "max_count": 4, "per_count": 0.03} | 通用增伤 | 其他增伤 |
| buffs.json | 明音凛E觉·传导强化 | `character_akane_e_area` | `action_dmg` | 0.3 | 通用增伤 | 动作增伤 |
| buffs.json | 明音凛·传导普攻 | `character_akane_transmit_area` | `action_dmg` | 0.6 | 通用增伤 | 动作增伤 |
| buffs.json | 月的对跖点·变轨增伤 | `arc_akane_skill` | `skill_dmg` | 0.28 | 其他增伤 | 动作增伤 |
| buffs.json | 月的对跖点·普攻增伤 | `arc_akane_basic` | `basic_dmg` | 0.28 | 其他增伤 | 动作增伤 |
| buffs.json | 月的对跖点·相伤 | `arc_akane_element` | `element_dmg_相` | 0.2 | 属性增伤 | 属性增伤 |
| buffs.json | 欧拉欧拉！：普伤叠层 | `arc_ora_basic_stack` | `basic_dmg` | 0.036 | 其他增伤 | 动作增伤 |
| buffs.json | 残虹1觉·永久狩 | `character_canhong_a1_hunt` | `other_dmg` | 0.4 | 其他增伤 | 其他增伤 |
| buffs.json | 残虹F觉·灭层蚀强化 | `character_canhong_f_annihilation_dot_layers` | `other_dmg` | {"action_last_hit_ratio": true, "effect_key": "other_dmg", "max_count": 20, "per_count": 0.12} | 通用增伤 | 其他增伤 |
| buffs.json | 残虹·狩 | `character_canhong_hunt` | `other_dmg` | 0.25 | 其他增伤 | 其他增伤 |
| buffs.json | 永恒圆舞曲：Q 后潜意伤 | `arc_waltz_q_mind` | `mind_dmg` | 0.1 | 其他增伤 | 属性增伤 |
| buffs.json | 浔1觉·欣欣渐行 | `character_xun_q_record_damage` | `action_dmg` | {"factor": 1, "source": "record_damage"} | 通用增伤 | 动作增伤 |
| buffs.json | 浔2觉·忆昔告远 | `character_xun_q_time_stop_damage` | `action_dmg` | 0.2 | 通用增伤 | 动作增伤 |
| buffs.json | 浔5觉·单目标全队增伤 | `character_xun_a5_single_enemy_damage` | `all_dmg` | 0.15 | 通用增伤 | 通用增伤 |
| buffs.json | 海月3觉·华彩终结增伤 | `character_haiyue_a3_huacai_q_damage` | `ultimate_dmg` | 0.2 | 其他增伤 | 动作增伤 |
| buffs.json | 海月：进入华彩后全队魂伤 | `character_haiyue_huacai_team_soul` | `element_dmg_魂` | 0.08 | 属性增伤 | 属性增伤 |
| buffs.json | 灵可·万籁同频 | `character_lingke_six_node_resonance_element_damage` | `element_dmg_咒` | 0.3 | 属性增伤 | 属性增伤 |
| buffs.json | 灵可·万籁同频 | `character_lingke_six_node_resonance_element_damage` | `element_dmg_灵` | 0.3 | 属性增伤 | 属性增伤 |
| buffs.json | 灵可·覆纹弱点感应 | `character_lingke_fuwen_follow_damage` | `follow_dmg` | 0.1 | 其他增伤 | 动作增伤 |
| buffs.json | 灵可·谐频同频强化 | `character_lingke_harmonic_joint_damage` | `other_dmg` | {"base_value": 1.5, "effect_key": "other_dmg", "max_amount": 100, "max_extra": 0.4, "per_point": 0.004, "resource": "谐频"} | 通用增伤 | 其他增伤 |
| buffs.json | 灵可·超负荷共鸣负面增伤 | `character_lingke_q_burst_negative_damage` | `other_dmg` | {"count_def_down": true, "count_res_down": true, "damage_tags": ["DOT"], "effect_key": "other_dmg", "include_all_enemy_debuffs": true, "max_count": 5, "per_count": 0.04} | 通用增伤 | 其他增伤 |
| buffs.json | 现实避难所：Q 后附着伤害 | `arc_reality_shelter_q_attach` | `attach_dmg` | 0.1 | 其他增伤 | 动作增伤 |
| buffs.json | 白藏1觉·噤后下一技能增伤 | `character_baizang_a1_next_skill` | `action_dmg` | {"factor": 1, "source": "word_damage"} | 通用增伤 | 动作增伤 |
| buffs.json | 真红3觉共鸣 | `character_zhenhong_q1_self_all_dmg` | `all_dmg` | 0.3 | 通用增伤 | 通用增伤 |
| buffs.json | 穿过胭红蜃景满：Q 后光伤穿防 | `arc_crimson_mirage_full_q_light_def` | `element_dmg_光` | 0.512 | 属性增伤 | 属性增伤 |
| buffs.json | 穿过胭红蜃景：Q 后光伤穿防 | `arc_crimson_mirage_q_light_def` | `element_dmg_光` | 0.32 | 属性增伤 | 属性增伤 |
| buffs.json | 翳3觉·单一锁定目标增伤 | `character_yi_a3_single_locked_target_damage` | `all_dmg` | 0.08 | 通用增伤 | 通用增伤 |
| buffs.json | 翳4觉·兽牙附着强化 | `character_yi_a4_attachment_state` | `attach_dmg` | {"effect_key": "attach_dmg", "interval_ticks": 10, "max_value": 0.4, "per_interval": 0.02} | 其他增伤 | 动作增伤 |
| buffs.json | 翳·终结后兽牙强化 | `character_yi_q_beast_damage` | `action_dmg` | 0.1 | 通用增伤 | 动作增伤 |
| buffs.json | 薄荷2觉·变轨强化 | `character_mint_a2_e_damage` | `action_dmg` | 0.4 | 通用增伤 | 动作增伤 |
| buffs.json | 薄荷6觉·低血目标增伤 | `character_mint_a6_low_hp_damage` | `all_dmg` | 0.15 | 通用增伤 | 通用增伤 |
| buffs.json | 达芙蒂尔·幻影移行强化 | `character_daphne_shift_damage` | `action_dmg` | 0.8 | 通用增伤 | 动作增伤 |
| buffs.json | 鉴定师4觉·终结增伤上限 | `character_protagonist_a4_q_damage` | `action_dmg` | 0.25 | 通用增伤 | 动作增伤 |
| buffs.json | 鉴定师·极轨终结增伤 | `character_protagonist_q_damage` | `ultimate_dmg` | 0.25 | 其他增伤 | 动作增伤 |
| buffs.json | 错误的门：治疗后其他队友通用增伤 | `arc_wrong_door_heal_team_damage` | `all_dmg` | 0.15 | 通用增伤 | 通用增伤 |
| buffs.json | 错误的门：治疗后灵伤 | `arc_wrong_door_heal_spirit` | `element_dmg_灵` | 0.3 | 属性增伤 | 属性增伤 |
| buffs.json | 阿德勒：明悟之慧佑咒伤 | `character_adler_e_team_curse` | `element_dmg_咒` | 0.1 | 属性增伤 | 属性增伤 |
| buffs.json | 静谧山庄：普攻心灵伤叠层 | `cartridge_mountain_basic_mind_stack` | `mind_dmg` | 0.12 | 其他增伤 | 属性增伤 |
| buffs.json | 预备备：EQ 后普攻闪反增伤 | `arc_ready_eq_basic_dodge` | `basic_dmg` | 0.15 | 其他增伤 | 动作增伤 |
| buffs.json | 预备备：EQ 后普攻闪反增伤 | `arc_ready_eq_basic_dodge` | `dodge_counter_dmg` | 0.15 | 其他增伤 | 动作增伤 |
| buffs.json | 预备备：司令虎符对 Boss 增伤 | `arc_ready_tally_boss_damage` | `all_dmg` | 0.1 | 通用增伤 | 通用增伤 |
| buffs.json | 黑羽D觉·全队攻击与通伤提升 | `character_heiyu_d_attack` | `all_dmg` | 0.12 | 通用增伤 | 通用增伤 |
| buffs.json | 黑羽F觉·魂伤提升 | `character_heiyu_f_soul` | `element_dmg_魂` | 0.25 | 属性增伤 | 属性增伤 |
| cartridges.json | 三型散件 | `cartridge_7983d1fbd7` | `element_dmg_光` | 0.1 | 属性增伤 | 属性增伤 |
| cartridges.json | 三型散件 | `cartridge_7983d1fbd7` | `element_dmg_咒` | 0.1 | 属性增伤 | 属性增伤 |
| cartridges.json | 三型散件 | `cartridge_7983d1fbd7` | `element_dmg_暗` | 0.1 | 属性增伤 | 属性增伤 |
| cartridges.json | 三型散件 | `cartridge_7983d1fbd7` | `element_dmg_灵` | 0.1 | 属性增伤 | 属性增伤 |
| cartridges.json | 三型散件 | `cartridge_7983d1fbd7` | `element_dmg_相` | 0.1 | 属性增伤 | 属性增伤 |
| cartridges.json | 三型散件 | `cartridge_7983d1fbd7` | `element_dmg_魂` | 0.1 | 属性增伤 | 属性增伤 |
| cartridges.json | 失落光芒 | `cartridge_f4282fad3f` | `element_dmg_光` | 0.1 | 属性增伤 | 属性增伤 |
| cartridges.json | 恶魔之血：诅咒 | `cartridge_bb71141f17` | `all_dmg` | 0.18 | 通用增伤 | 通用增伤 |
| cartridges.json | 恶魔之血：诅咒 | `cartridge_bb71141f17` | `element_dmg_魂` | 0.1 | 属性增伤 | 属性增伤 |
| cartridges.json | 森林萤火之心 | `cartridge_29793225a0` | `element_dmg_灵` | 0.1 | 属性增伤 | 属性增伤 |
| cartridges.json | 真红：双生蝶 | `cartridge_505ea4a58f` | `element_dmg_咒` | 0.1 | 属性增伤 | 属性增伤 |
| cartridges.json | 街头拳王 | `cartridge_9229a5376c` | `element_dmg_相` | 0.1 | 属性增伤 | 属性增伤 |
| cartridges.json | 迪亚波罗斯 | `cartridge_2f40014b04` | `element_dmg_暗` | 0.1 | 属性增伤 | 属性增伤 |
| cartridges.json | 静谧山庄 | `cartridge_1d257dca56` | `mind_dmg` | 0.1 | 其他增伤 | 属性增伤 |
| formula_constants.json | 配装词条与默认面板 | `build` | `all_dmg` | {"kind": "percent", "label": "通伤", "unit_value": 0.01} | 通用增伤 | 通用增伤 |
| shaft_engine.js | 增伤赐福 | `blessing` | `all_dmg` | 0.15 | 通用增伤 | 通用增伤 |
| shaft_engine.js | 灵可E觉·无法环合增伤 | `lingke_joint` | `other_dmg` | 0.5 | 通用增伤 | 其他增伤 |

条目数按文件、规则与字段去重（不是独立乘区或可同时叠加数量）：其他增伤 7，动作增伤 44，属性增伤 51，通用增伤 30。

重新生成：`python3 scripts/audit_shaft_damage_bonuses.py > docs/shaft-damage-bonus-classification.md`。
