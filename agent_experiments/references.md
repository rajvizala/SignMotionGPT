# Annotated References

This file is the short reference map for the experiment code in `agent_experiments/src`.

## Sign-language papers

1. Duarte et al., "How2Sign: A Large-scale Multimodal Dataset for Continuous American Sign Language," CVPR 2021.
   - Why it matters: defines the sentence-level target domain and highlights the multimodal complexity of continuous ASL.
   - URL: https://openaccess.thecvf.com/content/CVPR2021/html/Duarte_How2Sign_A_Large-Scale_Multimodal_Dataset_for_Continuous_American_Sign_Language_CVPR_2021_paper.html

2. Zuo et al., "A Simple Baseline for Spoken Language to Sign Language Translation with 3D Avatars," ECCV 2024.
   - Why it matters: supports the idea that dictionary-like composition is a valid sign-production direction.
   - URL: https://arxiv.org/abs/2401.04730

3. Fang et al., "SignLLM: Sign Languages Production Large Language Models," ICCV Workshop 2025.
   - Why it matters: shows LLM-style sign production is now a valid research direction, but not necessarily solved for ASL motion generalization.
   - URL: https://openaccess.thecvf.com/content/ICCV2025W/CV4A11y/html/Fang_SignLLM_Sign_Language_Production_Large_Language_Models_ICCVW_2025_paper.html

## Motion-token and text-to-motion papers

4. Zhang et al., "T2M-GPT: Generating Human Motion from Textual Descriptions with Discrete Representations," CVPR 2023.
   - Why it matters: validates the exact VQ-VAE + autoregressive generator pattern and motivates corruption-based robustness.
   - URL: https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Generating_Human_Motion_From_Textual_Descriptions_With_Discrete_Representations_CVPR_2023_paper.html

5. Jiang et al., "MotionGPT: Human Motion as a Foreign Language," NeurIPS 2023.
   - Why it matters: motivates treating motion as a language and using prompt-style multitask learning.
   - URL: https://arxiv.org/abs/2306.14795

6. Guo et al., "MoMask: Generative Masked Modeling of 3D Human Motions," CVPR 2024.
   - Why it matters: motivates masked motion recovery as a strong semantic robustness signal.
   - URL: https://arxiv.org/abs/2312.00063

7. Kong et al., "Priority-Centric Human Motion Generation in Discrete Latent Space," ICCV 2023.
   - Why it matters: argues that not all motion tokens are equally important, which supports motion-aware regularization ideas.
   - URL: https://arxiv.org/abs/2308.14480

8. Li et al., "LaMP: Language-Motion Pretraining for Motion Generation, Retrieval, and Captioning," ICLR 2025.
   - Why it matters: motivates stronger language-motion alignment instead of relying on generic text embeddings.
   - URL: https://arxiv.org/abs/2410.07093

9. Zhang et al., "KMM: Key Frame Mask Mamba for Extended Motion Generation," 2024.
   - Why it matters: supports fine-grained alignment and robustness for long motion generation.
   - URL: https://arxiv.org/abs/2411.06481

## Sequence-modeling papers

10. Ruoss et al., "Randomized Positional Encodings Boost Length Generalization of Transformers," ACL 2023.
    - Why it matters: directly motivates randomized positional offsets in long and variable-length motion generation.
    - URL: https://aclanthology.org/2023.acl-short.161/

11. Rubin and Berant, "Retrieval-Pretrained Transformer: Long-range Language Modeling with Self-retrieval," TACL 2024.
    - Why it matters: motivates integrating retrieval into training rather than only using parametric memory.
    - URL: https://aclanthology.org/2024.tacl-1.66/
