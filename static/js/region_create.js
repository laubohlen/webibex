document.addEventListener("DOMContentLoaded", function () {
    // initilize leaflet map
    var map = L.map("map").setView([45.578559, 7.230163], 14);

    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 19,
        attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>',
    }).addTo(map);

    // display default marker and region circle
    var marker = L.marker([45.578559, 7.230163]).addTo(map);
    var circle = L.circle([45.578559, 7.230163], {
        fillOpacity: 0.1,
        radius: 2000,
    }).addTo(map);

    // move marker and region to clicked map location
    function onMapClick(event) {
        lat = event.latlng.lat;
        lng = event.latlng.lng;
        marker.setLatLng([lat, lng]).addTo(map);
        circle.setLatLng([lat, lng]).addTo(map);
    }

    map.on("click", onMapClick);

    // Update circle radius dynamically
    document.getElementById("radius").addEventListener("input", function () {
        let newRadius = parseFloat(this.value);
        if (!isNaN(newRadius) && newRadius > 0) {
            circle.setRadius(newRadius);
        }
    });

    // Prefill the input field with the initial radius value
    document.getElementById("radius").value = circle.getRadius();

    // parse values
    document.getElementById("region-form").addEventListener("submit", function (event) {
        let regionName = document.getElementById("region-name").value.trim();
        let radius = circle.getRadius();
        let latitude = marker.getLatLng().lat;
        let longitude = marker.getLatLng().lng;

        // Check if the region name is empty
        if (regionName === "") {
            alert("Region name is required!"); // Notify the user
            event.preventDefault(); // Stop form submission
            return;
        }

        // Assign values to hidden input fields
        document.getElementById("form-region-name").value = regionName;
        document.getElementById("form-radius").value = radius;
        document.getElementById("form-latitude").value = latitude;
        document.getElementById("form-longitude").value = longitude;
    });
});
