// main.js

import { loadArticle } from "../utils/utils.js";

const articleDescriptionsAndDates = [
    {
        description: "SSVM",
        date: "2025-02-10"
    },
    {
        description: "launch",
        date: "2025-02-10"
    }
]

const articles = articleDescriptionsAndDates.map(
    article => {
        return `news/${article.date}_${article.description}.html`
    }
)

const newsContainer = document.getElementById('news')

for (const article of articles) {
    await loadArticle(article, newsContainer);
}