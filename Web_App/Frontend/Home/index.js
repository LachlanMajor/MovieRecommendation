const toggleButton = document.getElementById('toggle-btn')
const sidebar = document.getElementById('sidebar')

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

function changePoster(button){
    const slider = document.querySelector('.slider');
    const slideWidth =  slider.offsetWidth;

    if (button.id === 'leftSlide') {
        slider.scrollBy({ left: -slideWidth, behavior: 'smooth'})
    } else {
        slider.scrollBy({ left: slideWidth, behavior: 'smooth'})
    }

}