// utils.js

export { loadArticle }

async function loadArticle(url, container) {
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`Failed to fetch ${url}`);
        const content = await response.text();
        const articleDiv = document.createElement('div');
        articleDiv.innerHTML = content;
        container.appendChild(articleDiv);
    } catch (error) {
        console.error(error);
        container.innerHTML += `<p>Error loading article: ${url}</p>`;
    }
}