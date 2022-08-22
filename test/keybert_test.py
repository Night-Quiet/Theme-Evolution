import spacy
from keybert import KeyBERT

a = "Purpose - The purpose of this paper is to comment on Steven Laporte's review of About and on Behalf of Scriptum Est by Suominen with the aim of clarifying conceptual confusions related to the notion of constitutive and the notion of value-in-itself in the review. Design/methodology/approach - The notion of constitutive as it appears in the reviewed monograph and Laporte's reasoning around the notion of value-in-itself as challenges to are discussed and their differences are analyzed. Findings - The notion of value-in-itself appears problematic as the reviewed monograph already claims. The notion of constitutive provides us with a more plausible foundation for challenging the exclusively instrumentality-based views of the rationality of the practice of the library and librarianship. Compared to the notion of constitutive as used here, the notions used by Laporte remain abstract. Originality/value - The notion of constitutive could be a key notion opening a perspective for conceiving of the historical, cultural, social, and political conditions of being of the humans as the foundation of the rationality of the library and librarianship."
nlp = spacy.load("en_core_web_lg")

kw_model = KeyBERT()
keywords = kw_model.extract_keywords(a, keyphrase_ngram_range=(1, 1), top_n=4, highlight=True)

for word, score in keywords:
    print(word)


