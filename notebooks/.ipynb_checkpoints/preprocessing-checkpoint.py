def clear_sentences(data):
    """
        Cleaning sentences, removing special characters and articles
    """
    
    sentences = list()
    for record in data:
        sentence = record['reviewText']
        sentence = sentence.lower()
        for char in "?.!/;:,":
            sentence = sentence.replace(char, '')

        sentence = sentence.split(sep=' ')
        sentence = [word for word in sentence if len(word) > 1]
        sentences.append(sentence)
        
    return sentences