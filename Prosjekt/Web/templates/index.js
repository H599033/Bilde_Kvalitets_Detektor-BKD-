$(document).ready(function () {
        $('#arrowButton').on('click', function () {
            $('#searchBar').toggle();
        });
        $('#sortForm').on('submit', function (event) {
            event.preventDefault();
            popup.close();
        });
    $('#filterDropdown').on('change', function () {
        var value = $(this).val().toLowerCase();
        $("table tr").each(function (index) {
            if (index !== 0) {
                $row = $(this);

                var motionBlur = $row.find("td:eq(4)").text().indexOf("Motion blur: ✖") > -1;
                var lavBelysning = $row.find("td:eq(4)").text().indexOf("Lav belysning: ✖") > -1;
                var urentKamera = $row.find("td:eq(4)").text().indexOf("Skittent kamera: ✖") > -1;

                if (value === "home") {
                    $row.show();
                } else if ((value === "motion_blur" && motionBlur) ||
                    (value === "lav_belysning" && lavBelysning) ||
                    (value === "urent_kamera" && urentKamera)) {
                    $row.show();
                } else {
                    $row.hide();
                }
            }
        });
    });
    $.getJSON("/display", function (data) {
        var table = $("<table></table>");
        var headerRow = $("<tr></tr>");
        headerRow.append("<th>ID</th>");
        headerRow.append("<th>Sted</th>");
        headerRow.append("<th>Tid</th>");
        headerRow.append("<th>Dato</th>");
        headerRow.append("<th>Kvalitet</th>");
        headerRow.append("<th>Orginal Bilder</th>");
        headerRow.append("<th>Korrigerte Bilder</th>");
        table.append(headerRow);
        $.each(data, function (key, value) {
            var row = $("<tr></tr>");
            row.append("<td><p>" + value.ID + "</p></td>");
            row.append("<td><p>" + value.sted + "</p></td>");
            row.append("<td><p>" + value.tid + "</p></td>");
            row.append("<td><p>" + value.dato + "</p></td>");
            var qualityIndicator = "";
            if (Boolean(value.motion_blur) == true) {
                qualityIndicator += "Motion blur: " + "<p style='color:red;'>✖</p>";
            }
            if (Boolean(value.lav_belysning) == true) {
                qualityIndicator += "Lav belysning: " + "<p style='color:red;'>✖</p>";
            }
            if (Boolean(value.urent_kamera) == true) {
                qualityIndicator += "Skittent kamera: " + "<p style='color:red;'>✖</p>";
            }
            if (qualityIndicator === "") {
                qualityIndicator = "Kvalitet: " + "<p style='color:green;'>✔</p>";
            }
            row.append("<td>" + qualityIndicator + "</td>");


            var imgCell = $("<td></td>");
            var select = $("<select class='image-selector'></select>");
            $.each(value.orginal_bilder, function (i, img) {
                select.append("<option value='" + img + "'>Image " + (i + 1) + "</option>");
            });
            imgCell.append(select);
            imgCell.append("<img class='selected-image' src='" + value.orginal_bilder[0] + "' alt='Orginal Bilder'>");
            row.append(imgCell);
            var redImgCell = $("<td></td>");
            var redSelect = $("<select class='image-selector'></select>");
            $.each(value.redigerte_bilder, function (i, img) {
                redSelect.append("<option value='" + img + "'>Image " + (i + 1) + "</option>");
            });
            redImgCell.append(redSelect);
            redImgCell.append("<img class='selected-image' src='" + value.redigerte_bilder[0] + "' alt='Ok'>");
            row.append(redImgCell);
            table.append(row);
        });
        $("#content").append(table);
    });
});
$(document).on('change', '.image-selector', function () {
    var selectedImage = $(this).val();
    $(this).siblings('.selected-image').attr('src', selectedImage);
});