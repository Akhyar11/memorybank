#!/bin/bash
export GROQ_API_KEY="YOUR_GROQ_API_KEY_HERE"

echo "Starting 5000 episodes..."
python memorybench_groq_generator.py --episodes 5000 --output data/memorybench_train.jsonl --difficulty 3 --augmentation 10
echo "Finished 5000 episodes."

echo "Starting 1000 episodes..."
python memorybench_groq_generator.py --episodes 1000 --output data/memorybench_hard.jsonl --difficulty 5 --augmentation 10
echo "Finished 1000 episodes."
