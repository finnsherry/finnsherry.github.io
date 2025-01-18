// main.js

import { loadArticle } from "../utils/utils.js";

const articleTitlesAndTypes = [
    // Journal
    // Conference
    // Preprints
    { 
        title: 'Diffusion-Shock_Filtering_on_the_Space_of_Positions_and_Orientations',
        type: 'preprints'
    },
    { 
        title: 'Crossing-Preserving_Geodesic_Tracking_on_Spherical_Images',
        type: 'preprints'
    },
    // Theses
    {
        title: 'Self-Healing_of_Comb_Polymer_Vitrimers',
        type: 'theses'
    }
];

const articles = articleTitlesAndTypes.map(
    article => {
        return {
            url: `publications/${article.type}/${article.title}.html`,
            container: document.getElementById(article.type)
        }
    }
)

function addToggleEvent(container) {
    container.addEventListener('click', function (event) {
        const publication = event.target.closest('.publication');
        if (publication) {
            const summary = publication.querySelector('.publication-summary');
            if (summary) {
                summary.style.display = summary.style.display === 'none' || !summary.style.display ? 'block' : 'none';
            }
        }
    });
}

for (const { url, container } of articles) {
    await loadArticle(url, container);
}

addToggleEvent(document.getElementById('preprints'));
addToggleEvent(document.getElementById('theses'));