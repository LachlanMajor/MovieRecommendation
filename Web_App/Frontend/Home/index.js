const toggleButton = document.getElementById('toggle-btn')
const sidebar = document.getElementById('sidebar')
const slider = document.querySelector('.slider');
const nav = document.getElementById('sliderNav')
const posters = slider.querySelectorAll('img');
const background = document.body;
const filePath = "https://image.tmdb.org/t/p/w500";
const filePath_back = "https://image.tmdb.org/t/p/original";

let movieImages = Array.from({ length: 20 }, () => ({
    poster: "/3nPwMd3KviJWaHzG9fZCqlwWMas.jpg",
    backdrop: "/e3hG3uadtcP0pYdRa5ch4ysQW76.jpg"
}));

function toggleSidebar(){
    sidebar.classList.toggle('close')
    toggleButton.classList.toggle('rotate')
    
    Array.from(sidebar.getElementsByClassName('show')).forEach(ul => {
        ul.classList.remove('show')
        ul.previousElementSibling.classList.remove('rotate')
    })

}

function toggleSubMenu(button){
    button.nextElementSibling.classList.toggle('show')
    button.classList.toggle('rotate')
    if(sidebar.classList.contains('close')){
        sidebar.classList.toggle('close')
        toggleButton.classList.toggle('rotate')
    }
}

document.addEventListener('click', function(event) {
    if (!sidebar.classList.contains('close') && !sidebar.contains(event.target)) {
        sidebar.classList.toggle('close')
        toggleButton.classList.toggle('rotate')

        Array.from(sidebar.getElementsByClassName('show')).forEach(ul => {
            ul.classList.remove('show')
            ul.previousElementSibling.classList.remove('rotate')
        })
    }
})

function posterSwap(button){   
    const slideWidth =  slider.clientWidth;
    if (button.id === 'leftSlide') {
        slider.scrollBy({ left: -slideWidth, behavior: 'smooth'});
    } else {
        slider.scrollBy({ left: slideWidth, behavior: 'smooth'});
    }
}

slider.addEventListener("scrollend", updateBackground);

function updateBackground() {
    const slideWidth =  slider.clientWidth;
    let currentIndex = Math.round(slider.scrollLeft / slideWidth);

    if (movieImages[currentIndex] && movieImages[currentIndex].backdrop) {
        const newBackdrop = movieImages[currentIndex].backdrop;
        background.style.backgroundImage = `url(${filePath_back + newBackdrop})`;
    }
    else{
        console.log(`Failed to update background at index: ${currentIndex}`)
    }
}

async function changePosters(recommendations) {
    let index = 0;
    const slideWidth =  slider.clientWidth;
    const currentIndex = (slideWidth > 0 && slider.scrollLeft > 0) ? Math.round(slider.scrollLeft / slideWidth) : 0;

    slider.innerHTML = '';
    nav.innerHTML = '';

    for (const recommendation of recommendations) {
        movie = recommendation['title'];
        year = recommendation['year'];
        id = recommendation['tmdbId']
        
        const img = document.createElement('img');
        const slideId = `slide-${index+1}`;
        img.id = slideId;
        img.alt = movie;

        try {
            const response = await fetch(`http://localhost:3000/home/poster?movie=${encodeURIComponent(movie)}&year=${year}&id=${id}`);
            const data = await response.json();

            if (data.posterPath) {
                img.src = `${filePath + data.posterPath}`;
            }

            movieImages[index] = {
                poster: data.posterPath || movieImages[index].poster,
                backdrop: data.backdropPath || movieImages[index].backdrop
            };

        } catch (err) {
            img.src = "https://image.tmdb.org/t/p/w500/3nPwMd3KviJWaHzG9fZCqlwWMas.jpg"
            console.error('Error fetching poster:', err);
        }

        slider.appendChild(img);

        const anchor = document.createElement('a');
        anchor.href = `#${slideId}`
        nav.appendChild(anchor)

        index++;
    }

    document.querySelector('.slider-wrapper').classList.remove('loading');
    background.style.backgroundImage = `url(${filePath_back + movieImages[currentIndex].backdrop})`;
}

function getRecommendations() {
    const checkedBoxes = document.querySelectorAll('input[type="checkbox"]:checked');
    const checkedValues = Array.from(checkedBoxes).map(cb => cb.id);

    const yearRanges = document.querySelectorAll('input[type="number"]');
    const years = Array.from(yearRanges).map(y => y.value);

    fetch('http://localhost:3000/home', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ genres: checkedValues, years: years})
    })
    .then(response => {
        if(!response.ok){
            throw new Error(`Server error: ${response.status} ${response.statusText}`)
        }
        return response.json()
    })
    .then(data =>{
        changePosters(data['message']);
    })
    .catch(err => {
        console.error(err)
    })
}
