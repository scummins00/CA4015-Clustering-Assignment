# CA4015-Clustering-Assignment
This repository is to store the Jupyter Book and accompanying files for the Clustering Assignment as part of CA4015, Advanced Machine Learning.
Find the repository [here](https://github.com/scummins00/CA4015-Clustering-Assignment).

## Jupyter Book
This project is made accessible through Jupyter Book. Access it [here](https://scummins00.github.io/CA4015-Clustering-Assignment/intro.html).
If you're having difficulty using Jupyter Book, please refer to the [Building the Jupyter Book](instruction) section below.
## Required Packages
Please find all the packages required to run this book as intended in the `requirements.txt` file.

<a name='instruction'></a>## Building the Jupyter Book:
1. Make changes to your book in the `main` branch.
2. Rebuild the book with `jupyter-book build book/`
3. Use `ghp-import -n -p -f book/_build/html` to push the new pages to the gh-pages branch.

## Building a PDF
This Jupyter Book is available in pdf version. To create the pdf, you will need `pyppeteer` to convert HTML to PDF.

Install `pyppeteer` using the following command:
`pip install pyppeteer`

Once installed, you can build the PDF using:
`jupyter-book build book/ --builder pdfhtml`

Please find the book at: `book/_build/pdf/book`