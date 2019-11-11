#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
import string

def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated) 
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        
        """


    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        ### remove punctuation

        text_string = content[1].translate(str.maketrans("", "", string.punctuation))

        ### project part 2: comment out the line below
        # words = text_string

        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)
        from nltk.stem.snowball import SnowballStemmer
        stemmer = SnowballStemmer("english", ignore_stopwords=True)

        new_sentance = []

        for word in text_string.split():
            new_sentance.append(stemmer.stem(word))

        words = " ".join(new_sentance)





    return words

    
    #    from nltk.stem.snowball import SnowballStemmer
    #     from nltk.corpus import stopwords

    #     stemmer = SnowballStemmer("english")

    #     split_text = text_string.split(" ")
        
    #     sw = stopwords.words("english")
    #     new_sentance = []


        
    #     print(" ".join(new_sentance))
    #     print(split_text)

    # return words

    

def main():
    ff = open("../text_learning/test_email.txt", "r")
    text = parseOutText(ff)
    print(text)



if __name__ == '__main__':
    main()

