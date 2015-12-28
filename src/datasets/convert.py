import sys
import glob
def get_keyword(s, start, end):
    #return the array of keyword at index 0 and the rest content at index 1
    return s.split(start)[1].split(end)

def get_keywords(s, start, end, brace = False):
    ret = []
    tokens = s.split(start)
    for token in tokens[:len(tokens)-1]:
        tmp = token.split(end)
        if len(tmp) > 1 and brace:
            ret.append(tmp[0].split("<")[0])
        elif len(tmp) > 1:
            ret.append(tmp[0])
    tmp = tokens[-1].split(end)
    ret.append(tmp[0])
    return " ".join(filter(lambda a: a != " ",ret)), tmp[1]

def remove_brace(s):
    ret = []
    tokens = s.split("<")
    for token in tokens:
        tmp = token.split(">")
        if len(tmp) > 1:
            ret.append(tmp[1])
        else:
            ret.append(tmp[0])
    return " ".join(filter(lambda a: a != " ", ret))

def parse_data(filename):
    article = []
    ret = []
    with open(filename, "r") as fp:
        for line in fp:
            article.append(line.strip())
        plain = ' '.join(article)
        
        rest = plain
        ret.append("<title>")
        if plain.find("<docTitle>") != -1:
            title, rest = get_keywords(rest, "<docTitle>", "</docTitle>")
            ret.append(remove_brace(title))
        ret.append("</title>")
        
        ret.append("<author>")
        if rest.find("<docAuthor>") != -1:
            author, rest = get_keywords(rest, "<docAuthor>", "</docAuthor>")
            ret.append(remove_brace(author))
        ret.append("</author>")
            
        
        ret.append("<affiliation>")
        if rest.find("<affiliation>") != -1:
            affiliation, rest = get_keywords(rest, "<affiliation>", "</affiliation>", True)
            ret.append(remove_brace(affiliation))
        ret.append("</affiliaion>")

        ret.append("<address>")
        if plain.find("<address>") != -1:
            address, rest = get_keywords(plain, "<address>", "</address>")
            ret.append(remove_brace(address))
        ret.append("</address>")
    return ret

if __name__ == '__main__':
    fileset = []
    for file_path in glob.glob(sys.argv[1]+"/*"):
        fileset.append(file_path)
    writefile = open("grobid.tagged.txt", "w")
    for f in fileset:
        tokens = parse_data(f) 
        writefile.write(' '.join(tokens))
        writefile.write('\n')
    writefile.close()


