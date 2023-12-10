document.addEventListener('htmx:afterRequest', function(evt) {
    changeSelectedExpert();
});

function changeSelectedExpert() {
    const currentExpert = document.querySelector('#title').textContent.substring(4);
    const selectElement = document.querySelector('.select');
    const options = selectElement.options;

    // Loop through all the options and set defaultSelected
    for (let i = 0; i < options.length; i++) {
        if (options[i].value === currentExpert) {
            options[i].defaultSelected = true;
            options[i].selected = true;
        } else {
            options[i].defaultSelected = false;
        }
    }

}