# 二维医学图像工具调研

调研日期：2026-10-14。范围：单张二维 PNG/JPG，包括胸片、已渲染的 CT/MRI 切片、超声静态图、病理图块、皮肤和眼底照片。以下是文档和源码核对后的选型建议，尚未在本机下载权重或进行性能实测。

## 结论与优先顺序

建议给 Agent 增加独立的图像工具，但按影像类型和任务选择后端。PNG/JPG 是存储格式，无法据此确定医学模态；胸片分类模型不适用于 CT 截图或皮肤照片。

如果先做一个能复现的版本，建议选择胸片：基础图像检查与裁剪 + TorchXRayVision 异常分类，随后加入 MedSAM 分割。若核心需求是混合模态图像问答，可以另评估 MedGemma 1.5 4B。分割返回区域，分类返回固定标签分数，视觉问答返回生成文本，三类输出的含义应明确区分。

## 候选后端

### TorchXRayVision：胸片异常分类，优先接入

提供预训练胸片分类模型和部分分割模型。官方示例直接读取 JPG，也提供 PNG/JPG 加载路径；可封装 `chest_xray_findings(image_id)`，返回各异常标签及模型分数。CPU 可作为功能验证起点，吞吐量需实测。仓库代码为 Apache-2.0。

关键约束：只用于对应的胸片任务；固定权重版本、标签顺序和预处理。官方模型要求特定强度归一化范围，不能把普通 RGB 数值直接送入模型。未经校准的分数不应包装成个体患病概率。

来源：[仓库](https://github.com/mlmed/torchxrayvision)、[图像加载与预处理](https://github.com/mlmed/torchxrayvision/blob/main/docs/source/utilities.rst)、[常见归一化问题](https://github.com/mlmed/torchxrayvision/blob/main/docs/source/faq.rst)。

### MedSAM / LiteMedSAM：带框提示的二维分割，优先接入

官方推理脚本接受图片路径和目标框 `--box`，可封装 `segment_roi(image_id, box)`，返回 mask 和叠加图。它擅长在指定区域内分割目标；调用方仍需提供可靠的候选框。原版不是输入任意疾病名称就能自动定位的系统。仓库代码为 Apache-2.0。

LiteMedSAM 是同项目的轻量路线，适合资源有限时进一步验证。纯二维场景先看原版或 LiteMedSAM；MedSAM2 的重点是 3D 和视频。推理资源与速度须在实际设备上测量，不能把训练集群配置当成推理门槛。

来源：[MedSAM](https://github.com/bowang-lab/MedSAM)、[PNG 示例及推理脚本](https://github.com/bowang-lab/MedSAM/blob/main/MedSAM_Inference.py)、[LiteMedSAM](https://github.com/bowang-lab/MedSAM/blob/LiteMedSAM/README.md)。

### BiomedParse 原版：文字提示的多模态二维分割，第二阶段

支持多种医学成像类型，输入图像和文字对象描述后输出分割结果；可作为 `segment_by_text(image_id, target, modality)` 的候选后端。官方原版示例读取 PNG 并使用 CUDA；原始任务和训练资料以 PNG 为主。JPG 可在封装时解码为相同像素表示，但需要单独验证输入兼容性和压缩影响。

版本选择很重要：当前默认分支的 v2 重点面向 3D；原版二维实现可在 `main` 分支查看。官方说明新版专用二维预训练权重尚待发布，不能直接把 3D checkpoint 当作二维通用后端。项目还明确限定其模型的研究开发用途；部署前分别核对代码与权重条款。

来源：[当前仓库及版本说明](https://github.com/microsoft/BiomedParse)、[原版 main 分支](https://github.com/microsoft/BiomedParse/blob/main/README.md)、[官方模型](https://huggingface.co/microsoft/BiomedParse)。

### MedGemma 1.5 4B：医学图像问答，可选专家后端

Google 的官方模型卡包含二维医学图像理解评估，覆盖放射、皮肤、病理、眼科等场景。适合封装 `medical_image_qa(image_id, question)`，由主 Agent 请求一次独立的图像解读，再结合病例和文献推理。输出以文本为主，不提供与分割工具等价的像素掩膜。

建议独立部署并保持单次请求无额外会话状态，避免与主 Agent 的 history 混在一起。GPU 部署更实际；这里不提供未经实测的显存承诺。权重受 Health AI Developer Foundations 条款约束，下载、托管和商业使用须按模型条款核对。免费获取权重与免费托管 API 是不同的成本模式。

来源：[官方模型卡](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card)。

### CheXagent：胸片视觉问答，作为专科替代方案

适合胸片的描述、问答和特定异常询问，可作为上述问答工具的胸片专用后端。官方提供本地图像路径调用示例；`CheXagent-2-3b` 模型卡标注 MIT，仓库同时明确其研究用途。推理依赖、权重版本与图像处理方式需固定。

与 TorchXRayVision 相比，这是生成式专家；不能把生成文本当成独立的像素定位或已校准分类分数。若已经接了医学问答模型，应通过相同病例对照决定是否需要第二个问答后端。

来源：[官方仓库](https://github.com/Stanford-AIMI/CheXagent)、[3B 模型卡](https://huggingface.co/StanfordAIMI/CheXagent-2-3b)。

### BiomedCLIP：图文匹配、检索和粗分类，可选

适合 `medical_image_match(image_id, candidate_texts)` 或相似图检索。官方示例展示 JPG/PNG 图文零样本分类，模型许可证为 MIT。它需要候选文字或检索库，返回相似度/候选排序，适合辅助筛选影像类型，不能直接承担开放诊断，也没有分割输出。模型卡注明英语语料限制。

来源：[模型卡](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)、[许可证](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/blob/main/LICENSE.md)。

## 可以借鉴的医学 Agent 工具库

[MedRAX](https://github.com/bowang-lab/MedRAX) 与本项目需求很接近：组合胸片分类、分割、定位、视觉问答等工具。可以参考其工具封装、选择性加载和评估设计，将需要的后端适配为 AMA 的 `Tool`。其 README 对 MedSAM 同时存在能力介绍和“后续加入支持”的说明，因此不能只看功能清单就认定每个适配器都已实现。

## 针对当前 AMA 的接入建议

建议的工具接口（尚未实现）：

1. `inspect_image(image_id)`：图像尺寸、通道、位深、来源模态元数据和基本质量信息。
2. `crop_image(image_id, box)`：返回裁剪图、原图坐标和裁剪变换，用于观察局部。
3. `chest_xray_findings(image_id)`：胸片分类分数与明确的标签集合。
4. `segment_roi(image_id, box)`：分割 mask、叠加图和像素区域统计。
5. `medical_image_qa(image_id, question)`：可选的医学视觉专家调用。

输入建议使用当前 Episode 已释放的 evidence/image ID，在宿主端解析图片路径。工具应校验模态；未知模态、非胸片输入胸片分类器等情况应返回不适用，而不是强行生成结果。

输出建议包含后端模型版本、输入文件哈希、预处理参数、坐标系和单位。裁剪/缩放后的框和 mask 必须映射回原图。像素面积可由 mask 计算，但毫米或平方毫米测量需要可靠的像素间距/标尺；仅有 JPG/PNG 文件不足以保证这些信息存在。CT 截图也不能直接把显示灰度当成原始 HU。以上为工程约束，不是额外模型能力。

当前 `ToolResult` 已支持 JSON 文本、图片附件和 evidence ID，因此可以返回叠加图与结构化统计。`model.py` 已能编码图片，但最终是否能向主模型发送这些图像仍由服务端能力决定。若主模型只接收文本，可以读取结构化工具结果，但不能将该模式等同于主模型亲自检查过叠加图。

建议把 PyTorch、CUDA 和各模型的依赖放在独立推理进程/服务，AMA 只保留轻量工具适配器。这样可以单独缓存权重、控制显存，避免给当前标准库检索工具引入大型且互相冲突的依赖。

## 成本与验证

这些候选以下载权重后本地部署为主，不能当作现成且不限量的免费在线 API。自托管可以没有按次 API 费用，但仍有机器、显卡、存储和运维成本，权重条款也各不相同。

最小验证流程：固定一批同模态图片，先验证解码、错误输入拒绝、坐标映射和结果格式，再测分割 Dice/IoU、分类指标及单张耗时/显存。Agent 层比较同一模型和相同病例的“直接看图”与“看图 + 算法工具”，同时记录工具次数和失败率。只有接口调用成功不足以说明医疗图像分析质量提高。

本轮只做调研，没有安装模型或注册新的图像算法工具。
