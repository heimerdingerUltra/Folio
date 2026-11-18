"""Hand-written document-level metadata for the five papers.

``evidence`` / ``problem_anchors`` are substrings a chunk must contain for the
field to count as supported by that chunk (see ``grounded_training_examples``).
"""

GOLD_PAPER_METADATA: dict[str, dict] = {
    "Attention_Is_All_You_Need.pdf": {
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit",
                    "Llion Jones", "Aidan N. Gomez", "Łukasz Kaiser", "Illia Polosukhin"],
        "problem": ("Dominant sequence transduction models rely on recurrent or convolutional "
                    "networks, which limits parallelisation and makes training slow; the paper "
                    "asks whether a model based solely on attention can do better."),
        "problem_anchors": ["recurrent", "attention", "parallel"],
        "metrics": [
            {"text": "28.4 BLEU on WMT 2014 English-to-German", "evidence": ["28.4", "English-to-German"]},
            {"text": "41.8 BLEU on WMT 2014 English-to-French", "evidence": ["41.8", "English-to-French"]},
            {"text": "3.5 days of training on eight GPUs", "evidence": ["3.5 days"]},
        ],
    },
    "LLaMA.pdf": {
        "title": "LLaMA: Open and Efficient Foundation Language Models",
        "authors": ["Hugo Touvron", "Thibaut Lavril", "Gautier Izacard", "Xavier Martinet",
                    "Marie-Anne Lachaux", "Timothee Lacroix", "Baptiste Rozière", "Naman Goyal",
                    "Eric Hambro", "Faisal Azhar", "Aurelien Rodriguez", "Armand Joulin",
                    "Edouard Grave", "Guillaume Lample"],
        "problem": ("State-of-the-art large language models are trained on proprietary data and "
                    "are not accessible; the paper shows that competitive foundation models can be "
                    "trained using publicly available datasets exclusively."),
        "problem_anchors": ["publicly", "state-of-the-art"],
        "metrics": [
            {"text": "Models range from 7B to 65B parameters", "evidence": ["7B to 65B"]},
            {"text": "LLaMA-13B outperforms GPT-3 (175B) on most benchmarks",
             "evidence": ["LLaMA-13B", "GPT-3", "most benchmarks"]},
            {"text": "LLaMA-65B is competitive with Chinchilla-70B and PaLM-540B",
             "evidence": ["Chinchilla-70B", "PaLM-540B"]},
        ],
    },
    "LoRA.pdf": {
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "authors": ["Edward Hu", "Yelong Shen", "Phillip Wallis", "Zeyuan Allen-Zhu",
                    "Yuanzhi Li", "Shean Wang", "Lu Wang", "Weizhu Chen"],
        "problem": ("Full fine-tuning retrains all parameters of a pre-trained model, which becomes "
                    "infeasible as models grow because every downstream task needs its own full "
                    "copy of the weights."),
        "problem_anchors": ["fine-tuning", "less feasible"],
        "metrics": [
            {"text": "Trainable parameters reduced by 10,000 times", "evidence": ["10,000 times"]},
            {"text": "GPU memory requirement reduced by 3 times", "evidence": ["GPU memory requirement by 3 times"]},
            {"text": "No additional inference latency", "evidence": ["no additional inference latency"]},
        ],
    },
    "RAG.pdf": {
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors": ["Patrick Lewis", "Ethan Perez", "Aleksandra Piktus", "Fabio Petroni",
                    "Vladimir Karpukhin", "Naman Goyal", "Heinrich Küttler", "Mike Lewis",
                    "Wen-tau Yih", "Tim Rocktäschel", "Sebastian Riedel", "Douwe Kiela"],
        "problem": ("Pre-trained language models store knowledge in their parameters but cannot "
                    "easily access, update or attribute it, and lag behind task-specific "
                    "architectures on knowledge-intensive tasks."),
        "problem_anchors": ["knowledge-intensive", "parametric"],
        "metrics": [
            {"text": "State of the art on three open-domain QA tasks",
             "evidence": ["state of the art on three open"]},
            {"text": "44.5 Exact Match on Natural Questions (RAG-Sequence)",
             "evidence": ["44.5", "RAG-Seq"]},
        ],
    },
    "ChunkNorris.pdf": {
        "title": "ChunkNorris: A High-Performance and Low-Energy Approach to PDF Parsing and Chunking",
        "authors": ["Mathieu Ciancone", "Clovis Varangot-Reille", "Marion Schaeffer"],
        "problem": ("In retrieval-augmented generation, PDF parsing and chunking quality drives "
                    "retrieval quality, but accurate ML-based parsers are slow and energy-hungry; "
                    "the paper proposes a fast heuristic-based alternative."),
        "problem_anchors": ["parsing", "chunking", "heuristic"],
        "metrics": [
            {"text": "ChunkNorris parses a document in 105 ms on average", "evidence": ["ChunkNorris", "105 ms"]},
            {"text": "ChunkNorris consumes 0.47 Wh of CPU energy", "evidence": ["ChunkNorris", "0.47 Wh"]},
        ],
    },
}
