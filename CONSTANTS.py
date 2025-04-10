DATA_ENTITY_DICT = {
    "bc2gm": "gene/protein",
    "bc5chem": "chemical",
    "bc5disease": "disease/illness",
    "chemprotchem": "chemical",
    "chemprotgene": "gene/protein",
}

ENTITY2DEFINITION = {
    "gene/protein": "A gene is a fundamental unit of heredity that carries and transmits information from one generation to the next. Genes are segments of DNA (deoxyribonucleic acid), which is a molecule that contains the instructions an organism needs to develop, grow, and function. Each gene serves as a code for a specific protein or set of proteins, and the sequence of nucleotides in a gene determines the order in which amino acids are assembled to form these proteins.\nProteins are one of the most abundant organic molecules in living systems and have the most diverse range of functions of all macromolecules. Proteins may be structural, regulatory, contractile, or protective; they may serve in transport, storage, or membranes; or they may be toxins or enzymes. Each cell in a living system may contain thousands of proteins, each with a unique function. Their structures, like their functions, vary greatly. They are all, however, polymers of amino acids, arranged in a linear sequence.",
    "chemical": "Chemical refers to any substance having a distinct molecular composition that is produced by or used in a chemical process. Chemicals can be elements or compounds, and they can exist in various forms—solid, liquid, or gas.",
    "disease/illness": "A disease is a pathological condition of a part, organ, or system of an organism resulting from various causes, such as infection, genetic defect, or environmental stress. Diseases can manifest in a variety of ways, affecting the structure or function of organs or systems and often leading to specific signs and symptoms. They can be caused by microorganisms (such as bacteria, viruses, fungi, and parasites), genetic mutations, lifestyle factors, environmental exposures, or a combination of these.",
}

ENTITY2EXAMPLESFILE = {
    "gene/protein": f"genes.json",
    "chemical": f"chemicals.json",
    "disease/illness": f"diseases.json",
}

MODEL_NAME2HUGGINGFACE = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
    "gemma-2b-it": "google/gemma-2b-it",
    "gemma-7b-it": "google/gemma-7b-it",
    "llama2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
}
