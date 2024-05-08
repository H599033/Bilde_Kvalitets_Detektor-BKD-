$(document).ready(function () {
    $('#arrowButton').on('click', function () {
        $('#searchBar').toggle();
    });

    $('#resetButton').on('click', resetTable);
    $('#logo').on('click', resetTable);
    $('#kvalitet').on('change', filterTable);
    $('#startDate').on('change', filterTable);
    $('#endDate').on('change', filterTable);
    $('#startTime').on('change', filterTable);
    $('#endTime').on('change', filterTable);
    $('#id').on('change', filterTable);
    $('#place').on('change', filterTable);


    function filterTable() {
        var kvalitetValue = $('#kvalitet').val().toLowerCase();
        var startDateValue = $('#startDate').val() ? new Date($('#startDate').val()) : null;
        var endDateValue = $('#endDate').val() ? new Date($('#endDate').val()) : null;
        var startTimeValue = $('#startTime').val() ? $('#startTime').val() : null;
        var endTimeValue = $('#endTime').val() ? $('#endTime').val() : null;
        var idValue = $('#id').val() ? $('#id').val().toLowerCase() : null;
        var placeValue = $('#place').val().toLowerCase();

        $("table tr").each(function (index) {
            if (index !== 0) {
                $row = $(this);

                var motionBlur = $row.find("td:eq(4)").text().indexOf("Motion blur: ✔") > -1;
                var lavBelysning = $row.find("td:eq(4)").text().indexOf("Lav belysning: ✔") > -1;
                var vaatDekk = $row.find("td:eq(4)").text().indexOf("Vått dekk: ✔") > -1;
                var godkjent = $row.find("td:eq(4)").text().indexOf("Kvalitet: ✔") > -1;
                var rowDate = new Date($row.find("td:eq(3)").text());
                var rowTime = $row.find("td:eq(2)").text();
                var rowId = $row.find("td:eq(0)").text().toLowerCase();
                var rowPlace = $row.find("td:eq(1)").text().indexOf("Bergen") > -1;

                var kvalitetMatch =     
                    (kvalitetValue === "select..." ||
                    (kvalitetValue === "godkjent" && godkjent)||  
                    (kvalitetValue === "motion_blur" && motionBlur) ||
                    (kvalitetValue === "lav_belysning" && lavBelysning) ||
                    (kvalitetValue === "vaatt_dekk" && vaatDekk));

                var dateMatch = (!startDateValue || rowDate >= startDateValue) && (!endDateValue || rowDate <= endDateValue);
                var timeMatch = (!startTimeValue || rowTime >= startTimeValue) && (!endTimeValue || rowTime <= endTimeValue);
                var idMatch = (!idValue || rowId.includes(idValue));
                var placeMatch = (placeValue === "select..." || (!placeValue || (placeValue === "bergen" && rowPlace)));
                
                if (kvalitetMatch && dateMatch && timeMatch && idMatch && placeMatch) {
                    $row.show();
                } else {
                    $row.hide();
                }
          
            }
        });
    }

    // This will reset all the input fields in the form to their default values
    function resetTable() {
        $("table tr").show();
        $('#sortForm').find('input[type=search], input[type=time], input[type=date], input[type=text]').val('');
        $('#sortForm').find('select').prop('selectedIndex', 0);
    }
    $.getJSON("/display", function (data) {
        var table = $("<table></table>");
        var headerRow = $("<tr></tr>");
        headerRow.append("<th>ID</th>");
        headerRow.append("<th>Sted</th>");
        headerRow.append("<th>Tid</th>");
        headerRow.append("<th>Dato</th>");
        headerRow.append("<th>Kvalitet</th>");
        headerRow.append("<th>Original Bilder</th>");
        headerRow.append("<th>Korrigerte Bilder</th>");
        table.append(headerRow);
        $.each(data, function (key, value) {
            var row = $("<tr></tr>");
            row.append("<td><p>" + value.ID + "</p></td>");
            row.append("<td><p>" + value.sted + "</p></td>");
            row.append("<td><p>" + value.tid + "</p></td>");
            row.append("<td><p>" + value.dato + "</p></td>");
            var qualityIndicator = "";

            if(Boolean(value.motion_blur) == true || Boolean(value.lav_belysning) == true || Boolean(value.vaatt_dekk) == true){
                qualityIndicator += "Kvalitet: " + "<span style='color:red;'>✖</span>" + "<br>";
            }

            if (Boolean(value.motion_blur) == true) {
                qualityIndicator += "Motion blur: " + "<span style='color:green;'>✔</span>" + "<br>";
            }
            if (Boolean(value.lav_belysning) == true) {
                qualityIndicator += "Lav belysning: " + "<span style='color:green;'>✔</span>" + "<br>";
            }
            if (Boolean(value.vaatt_dekk) == true) {
                qualityIndicator += "Vått dekk: " + "<span style='color:green;'>✔</span>" + "<br>";
            }
            if (qualityIndicator === "") {
                qualityIndicator = "Kvalitet: " + "<span style='color:green;'>✔</span>";
            }
            row.append("<td>" + qualityIndicator + "</td>");
    
    
            var imgCell = $("<td></td>");
            var select = $("<select class='image-selector' style='position: absolute; top: 0; right: 0;'></select>");
            $.each(value.orginal_bilder, function (i, img) {
                select.append("<option value='" + img + "'>Image " + (i + 1) + "</option>");
            });
            var imgDiv = $("<div style='position: relative;'></div>");
            var image = $("<img class='selected-image' src='" + value.orginal_bilder[0] + "' alt='Original Bilder'>");
            var downloadLink = $("<a class='download-link' style='position: absolute; bottom: 0; right: 0;' href='" + value.orginal_bilder[0] + "' download>Download</a>");
            imgDiv.append(image);
            imgDiv.append(select);
            imgDiv.append(downloadLink);
            imgCell.append(imgDiv);
            row.append(imgCell);
    
            var redImgCell = $("<td></td>");
            var redSelect = $("<select class='image-selector' style='position: absolute; top: 0; right: 0;'></select>");
            $.each(value.redigerte_bilder, function (i, img) {
                redSelect.append("<option value='" + img + "'>Image " + (i + 1) + "</option>");
            });
            var redImgDiv = $("<div style='position: relative;'></div>");
            redImgDiv.append("<img class='selected-image' src='" + value.redigerte_bilder[0] + "' alt='Ok'>");
            redImgDiv.append(redSelect);
            redImgCell.append(redImgDiv);
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