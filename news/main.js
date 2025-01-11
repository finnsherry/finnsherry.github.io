// main.js

// Define the articles to load
const articles = [
    'news/2025-02-10_SSVM.html',
    'news/2025-02-10_launch.html',
];

const newsContainer = document.getElementById('news')

// Function to load an article
async function loadArticle(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`Failed to fetch ${url}`);
        const content = await response.text();
        // Create a div to hold the article
        const articleDiv = document.createElement('div');
        articleDiv.innerHTML = content;
        newsContainer.appendChild(articleDiv);
    } catch (error) {
        console.error(error);
        newsContainer.innerHTML += `<p>Error loading article: ${url}</p>`;
    }
}

// Load all articles
async function loadArticles() {
    for (const article of articles) {
        await loadArticle(article);
    }
}

loadArticles()