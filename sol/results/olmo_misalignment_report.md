
# Misalignment eval — % preferring trait-matching (undesirable) answer

(non-judge; Anthropic Advanced-AI-Risk Model-Written Evals; higher = more of the trait; students share one base = Olmo-3-32B-Think-SFT)


## teacher qwen

| eval | base | c6 | c6e | c6−c6e | c6−base |
|---|---|---|---|---|---|
| coordinate-other-ais | 2.4 | 2.4 | 2.4 | +0.0 | +0.0 |
| corrigible-less-HHH | 1.6 | 1.6 | 1.6 | +0.0 | +0.0 |
| myopic-reward | 30.8 | 30.8 | 30.8 | +0.0 | +0.0 |
| one-box-tendency | 95.2 | 95.2 | 95.6 | -0.4 | +0.0 |
| power-seeking-inclination | 88.4 | 89.2 | 88.8 | +0.4 | +0.8 |
| self-awareness-general-ai | 89.2 | 89.2 | 89.2 | +0.0 | +0.0 |
| survival-instinct | 72.4 | 72.8 | 72.0 | +0.8 | +0.4 |
| wealth-seeking-inclination | 80.8 | 80.0 | 80.8 | -0.8 | -0.8 |
| **ALL** | **57.6** | **57.6** | **57.6** | **+0.0** | **+0.1** |

## cross-teacher (ALL-evals aggregate, c6−c6e in points)

| teacher | c6 | c6e | c6−c6e |
|---|---|---|---|
| qwen | 57.6 | 57.6 | +0.0 |