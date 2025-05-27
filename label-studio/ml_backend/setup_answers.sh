export TRITON_MODEL=ensemble
export TRITON_ENDPOINT="http://triton:9000/v1"
export USE_INTERNAL_PROMPT_TEMPLATE=0
export DEFAULT_PROMPT=$(cat <<EOF
You are a medical expert. Answer the following question using only the information provided in the accompanying text. Follow these strict rules:

- Output only the final answer.
- Do not restate the question.
- Do not explain, elaborate, speculate, or add context.
- Do not add formatting, markdown, notes, or instructions.
- Only use content explicitly stated in the text.

Question: {text}
EOF
)
