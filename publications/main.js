// main.js

// Define the articles to load
const articles = [
    // Preprints
    { url: 'publications/preprints/Diffusion-Shock_Filtering_on_the_Space_of_Positions_and_Orientations.html', container: document.getElementById('preprints') },
    { url: 'publications/preprints/Crossing-Preserving_Geodesic_Tracking_on_Spherical_Images.html', container: document.getElementById('preprints') },
    // Theses
    { url: 'publications/theses/Self-Healing_of_Comb_Polymer_Vitrimers.html', container: document.getElementById('theses') }
];

// Function to load an article
async function loadArticle(url, container) {
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`Failed to fetch ${url}`);
        const content = await response.text();
        // Create a div to hold the article
        const articleDiv = document.createElement('div');
        articleDiv.innerHTML = content;
        container.appendChild(articleDiv);
    } catch (error) {
        console.error(error);
        container.innerHTML += `<p>Error loading article: ${url}</p>`;
    }
}

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

// Load all articles
articles.forEach(({ url, container }) => loadArticle(url, container));

// Make interactive
addToggleEvent(document.getElementById('preprints'));
addToggleEvent(document.getElementById('theses'));