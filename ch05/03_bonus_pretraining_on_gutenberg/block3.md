Block 3 · 对齐：RLHF / DPO / 偏好优化
时长建议 80 分钟：文档 35 → 习题 20 → 代码 25 配套代码：code/03_dpo_loss.py（CPU 可跑，从零实现 DPO 损失）


一、【文档】概念精讲
3.1 为什么 SFT 之后还要对齐？
SFT 教会模型"听指令、按格式答"，但它只学了"模仿示范回答"。问题：

很多任务没有唯一标准答案（"写一首诗""这个回答够不够有帮助"），难以用监督标签表达。
人类更容易比较两个回答哪个更好，而不是从零写出完美答案。

于是用人类偏好来对齐：给同一个 prompt 的两个回答标注"A 比 B 好"，让模型学会产出人更偏好的回答。 这就是 RLHF (Reinforcement Learning from Human Feedback) 和它的简化替代 DPO。
3.2 经典 RLHF 三步（必须能画出来）
第①步  SFT model（Block 2 已得到）              ← 起点，作为后续的初始策略 & 参考模型

          │

          ▼

第②步  训练 Reward Model (RM)

        数据：(prompt, 回答_chosen, 回答_rejected) 人类偏好对

        结构：在基座上加一个标量输出头（value head），输出"这个回答有多好"的分数 r

        损失：Bradley-Terry 排序损失

              L = -log σ( r(chosen) - r(rejected) )

        目标：让 RM 给 chosen 的分 > rejected 的分

          │

          ▼

第③步  用 PPO 优化策略模型，让它最大化 RM 给的奖励

        目标： maximize  E[ r(x,y) ]  −  β · KL( π_θ(y|x) ‖ π_ref(y|x) )

                          ↑ 让回答得高分        ↑ 别偏离 SFT 模型太远（防作弊/胡说/语言崩坏）

每一步在训什么：

第①步：训出一个会聊天的初始策略（已在 Block 2 完成）。
第②步：训一个"打分器"RM，它替代昂贵的实时人类标注。
第③步：策略模型生成回答→RM 打分→PPO 用这个分数当奖励信号更新策略。
3.3 三个你必须理解的关键概念
(1) KL 惩罚项：PPO 目标里那个 $-\beta \cdot KL(\pi_\theta | \pi_{ref})$。

$\pi_{ref}$ 是冻结的 SFT 模型（参考模型）。
作用：约束新策略别离原模型太远。没有它，模型会为了刷高 RM 分而胡言乱语 （比如疯狂重复 RM 喜欢的词、输出乱码但分数高）。
β 是平衡旋钮：β 大→更保守贴近 SFT；β 小→更激进追求高奖励但易崩。

(2) Reward Hacking（奖励作弊）：策略找到 RM 的漏洞，刷出高分但实际质量差。 例：RM 偏爱长回答 → 模型疯狂啰嗦；RM 偏爱礼貌词 → 模型满嘴"当然！很高兴帮您！"但不答正事。 这是 RLHF 最大痛点，靠 KL 约束 + 更好的 RM + 多样化数据缓解。

(3) PPO 为什么复杂：训练时同时需要 4 个模型在显存里： 策略模型(训练) + 参考模型(算 KL，冻结) + 奖励模型(打分，冻结) + 价值模型(估计优势，训练)。 工程上很重、调参敏感、不稳定。这正是 DPO 出现的动机。
3.4 DPO：把 RLHF 变成一个监督式 loss（重点吃透）
DPO (Direct Preference Optimization) 的洞见：RLHF 第②③步的最优解其实有闭式关系， 可以跳过"训 RM + 跑 PPO"，直接用偏好数据对策略模型做一个类似分类的损失：

$$\mathcal{L}{DPO} = -\log \sigma\left(\beta \log\frac{\pi\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)$$

其中 $y_w$=chosen(更好), $y_l$=rejected(更差)。直觉解读：

括号里是 chosen 的"相对参考模型的对数概率提升" 减去 rejected 的提升。
想让这个差变大 → 提高 chosen 的概率、压低 rejected 的概率，同时用 $\pi_{ref}$ 锚住别跑偏（KL 约束被隐式编码进去了）。
外面套 $-\log\sigma(\cdot)$ 就是 Bradley-Terry 排序损失，和 RM 的损失同形式，但作用在策略本身。

DPO vs RLHF/PPO：



RLHF (PPO)
DPO
需要单独训 RM？
要
不要
训练时模型数
4 个
2 个（策略 + 冻结参考）
需要在线采样生成？
要（on-policy）
不要（用离线偏好对，off-policy）
稳定性/易用性
难调、不稳
稳定、像普通监督训练
上限
理论更高（在线探索）
略低但实践常足够


一句话：DPO 用一个巧妙的损失函数，把"训奖励模型 + 跑强化学习"两步合并成一步监督式微调。 这也是为什么近两年很多开源对齐用 DPO（及其变体 IPO/KTO/ORPO/SimPO）而非完整 PPO。
3.5 数据：偏好对从哪来
(prompt, chosen, rejected)。来源：人类标注、AI 标注（RLAIF，用更强模型当裁判）、 规则构造（如用 SFT 模型采样多个回答，让裁判排序）。质量同样 >> 数量。


二、【自测习题】
概念题

为什么 SFT 不够、还需要偏好对齐？人类标注偏好比标注"完美答案"好在哪？
画出 RLHF 三步，并说出每步训练的是什么模型、用什么损失。
PPO 目标里的 KL 惩罚项作用是什么？去掉它会发生什么？β 调大/调小分别什么效果？
什么是 reward hacking？举两个具体例子，怎么缓解？
PPO 训练时显存里有哪 4 个模型？各自角色？
DPO 相比 PPO 省掉了哪两样东西？为什么它更稳定？
写出 DPO 损失公式，并解释"为什么最小化它等于提高 chosen、压低 rejected"。
DPO 里的参考模型 $\pi_{ref}$ 是什么？为什么需要它？

实操题 9. 在 code/03_dpo_loss.py 里把 β 调大，观察损失对"概率差"的敏感度变化。 10. 构造一个 chosen 概率 < rejected 概率的"标注错误"样本，看 loss 如何惩罚。


三、【代码】
python code/03_dpo_loss.py

该脚本从零实现 DPO 损失，并：

用一个玩具"策略/参考模型"（直接给 logprob）演示损失如何随 chosen/rejected 概率变化；
验证：当策略相对参考更偏好 chosen 时 loss 下降，反之上升；
同时实现 reward model 的 Bradley-Terry 损失，让你看清两者的同构关系。

真实 DPO 训练用 trl 库的 DPOTrainer，脚本末尾给了最小调用骨架（注释形式）。


四、习题答案
SFT 只会模仿示范，无法表达"哪个更好"这种相对、主观、无唯一答案的目标。人类比较两个回答比写出完美答案容易得多、也更可靠，能覆盖开放式任务。
①SFT：监督微调，next-token CE。②Reward Model：基座+标量头，Bradley-Terry 损失 $-\log\sigma(r_w-r_l)$。③PPO：策略模型，目标 $E[r]-\beta KL$。
KL 惩罚约束新策略别偏离 SFT 参考模型太远。去掉→模型为刷 RM 分而崩坏（重复、乱码、谄媚）。β 大更保守贴近 SFT；β 小更激进追奖励但易 reward hacking。
策略钻 RM 漏洞刷高分但质量差。例：RM 偏长→啰嗦；RM 偏礼貌→满嘴客套不答正题。缓解：KL 约束、更鲁棒的 RM、数据多样化、限制生成长度、定期用新数据重训 RM。
策略模型(训)、参考模型(冻结,算KL)、奖励模型(冻结,打分)、价值模型(训,估计优势/baseline)。
省掉：①单独训练 reward model；②在线强化学习采样(PPO)。更稳定因为它本质是离线监督式分类损失，没有 RL 的高方差采样和多模型耦合。
见 3.4 公式。最小化 $-\log\sigma(\Delta)$ 即最大化 $\Delta=\beta[\log\frac{\pi_\theta(y_w)}{\pi_{ref}(y_w)}-\log\frac{\pi_\theta(y_l)}{\pi_{ref}(y_l)}]$，要让它大就得提高 chosen 的相对对数概率、压低 rejected 的，$\pi_{ref}$ 作分母锚定防跑偏。
$\pi_{ref}$ 通常是冻结的 SFT 模型。需要它把"概率变化"相对化（隐式 KL 约束），防止策略为偏好而整体扭曲、保持语言能力。

