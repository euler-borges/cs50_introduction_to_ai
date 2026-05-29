import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    output = dict()
    page_number = len(corpus)

    # handling the case where `page` has no outgoing links
    if len(corpus[page]) == 0:
        damping_factor = 0
    
    # choosing a link at random from all pages in the corpus
    for it_page in corpus:
        output[it_page] = (1 - damping_factor) / page_number
    
    # choosing a link at random linked to by `page`
    for link in corpus[page]:
        output[link] += damping_factor / len(corpus[page])
    
    return output


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    output = dict()

    # picking random page to start with
    page = random.choice(list(corpus.keys()))

    for i in range(n):
        # updating the count for the page
        if page in output:
            output[page] += 1
        else:
            output[page] = 1
        
        # picking next page according to transition model
        model = transition_model(corpus, page, damping_factor)
        page = random.choices(
            population=list(model.keys()),
            weights=list(model.values()),
            k=1
        )[0]

    # normalizing the counts to get probabilities
    for page in output:
        output[page] /= n

    return output


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    ranks = dict()

    def calculate_pagerank(page):
        pagerank = (1 - damping_factor) / len(corpus)
        for other_page in corpus:
            if page in corpus[other_page]:
                pagerank += damping_factor * ranks[other_page] / len(corpus[other_page])
            elif len(corpus[other_page]) == 0:
                pagerank += damping_factor * ranks[other_page] / len(corpus)
        return pagerank

    # initializing PageRank values to 1 / N
    for page in corpus:
        ranks[page] = 1 / len(corpus)

    # iteratively updating PageRank values until convergence
    while True:
        new_ranks = dict()
        for page in corpus:
            new_ranks[page] = calculate_pagerank(page)
        
        # checking for convergence
        if all(abs(new_ranks[page] - ranks[page]) < 0.001 for page in corpus):
            break
        
        ranks = new_ranks

    return ranks

if __name__ == "__main__":
    main()
