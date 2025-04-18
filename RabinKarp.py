import math


def RabinKarp(text, ptn, d=256, prime=1009): # Take ASCII as example thus the default value of d is 256.
    l1 = len(ptn)
    l2 = len(text)
    h = math.pow(d, l1-1) % prime

    p_hash = 0 # Hash value of the pattern
    t_hash = 0 # Hash value of the window

    results = []

    # Initial hash value
    for i in range(l1):
        p_hash = (d * p_hash + ord(ptn[i])) % prime
        t_hash = (d * t_hash + ord(text[i])) % prime

    # Match by slide window
    for i in range(l2 - l1 + 1):
        if p_hash == t_hash:
            if text[i:i+l1] == ptn:
                results.append(i)

        if i < l2 - l1:
            t_hash = (d * (t_hash - ord(text[i]) * h) + ord(text[i+l1])) % prime
            if t_hash < 0:
                t_hash += prime

    return results


if __name__ == '__main__':
    text = ("I have a dream that one day this nation will rise up and live out the true meaning of its creed: \"We hold these truths to be self-evident, that all men are created equal.\""

    "I have a dream that one day on the red hills of Georgia, the sons of former slaves and the sons of former slave owners will be able to sit down together at the table of brotherhood."

    "I have a dream that one day even the state of Mississippi, a state sweltering with the heat of injustice, sweltering with the heat of oppression, will be transformed into an oasis of freedom and justice."

    "I have a dream that my four little children will one day live in a nation where they will not be judged by the color of their skin but by the content of their character."

    "I have a dream today!"

    "I have a dream that one day, down in Alabama, with its vicious racists, with its governor having his lips dripping with the words of \"interposition\" and \"nullification\" -- one day right there in Alabama little black boys and black girls will be able to join hands with little white boys and white girls as sisters and brothers."

    "I have a dream today!"

    "I have a dream that one day every valley shall be exalted, and every hill and mountain shall be made low, the rough places will be made plain, and the crooked places will be made straight; \"and the glory of the Lord shall be revealed and all flesh shall see it together.\""

    "This is our hope, and this is the faith that I go back to the South with."

    "With this faith, we will be able to hew out of the mountain of despair a stone of hope. With this faith, we will be able to transform the jangling discords of our nation into a beautiful symphony of brotherhood. With this faith, we will be able to work together, to pray together, to struggle together, to go to jail together, to stand up for freedom together, knowing that we will be free one day.")
    ptn = 'dream'
    print(RabinKarp(text, ptn, 256, 1009))