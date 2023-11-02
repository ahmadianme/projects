var plansPrices = {
    free: 0,
    start: 95000,
    movement: 225000,
    plus: 525000,
    growth: 625000,
    premier: 2225000,
    primier_free_trial: 0,
};

var planBasePrice = plansPrices.growth;
var extraLearnerPrice = 500;
var planBaseLearnerCount = 1200;

$(document).ready(function () {
    if ($("#learnerCount").length) {
        $("#learnerCount").ionRangeSlider();
        var learnerCount = $("#learnerCount").data("ionRangeSlider");
        learnerCount.update({
            min: '0',
            max: '4000',
            from_min: planBaseLearnerCount,
            from: planBaseLearnerCount,
            grid_num: '10',
            prettify_enabled: false,
            step: '10',
            grid: 'true',
            skin: 'theme',
            onChange: updateCustomizationPrice
        });
    }

    if ($("input[name='mainPlanDuration']").length) {
        setPlansPrices();

        $("input[name='mainPlanDuration']").change(function () {
            setPlansPrices();
        });
    }

    if ($("input[name='customizationPlanDuration']").length) {
        updateCustomizationPrice();

        $("input[name='customizationPlanDuration']").change(function () {
            updateCustomizationPrice();
        });
    }

    if ($('#pricing .view-more-button').length) {
        $('#pricing .view-more-button').click(function () {
            $('#pricing .view-more-hide-after').css('display', 'none');
            $('#pricing .view-more-content').css('display', 'block');

            return false;
        });
    }

    if ($("input[name='comparisonPlanDuration']").length) {
        $("input[name='comparisonPlanDuration']").change(function () {
            setPlansPrices();
        });
    }

    if ($("input[name='registerPlanDuration']").length) {
        $("input[name='registerPlanDuration']").change(function () {
            updateRegisterPrice();
        });
    }
});

function numberFormat(nStr) {
    nStr += '';
    x = nStr.split('.');
    x1 = x[0];
    x2 = x.length > 1 ? '.' + x[1] : '';
    var rgx = /(\d+)(\d{3})/;
    while (rgx.test(x1)) {
        x1 = x1.replace(rgx, '$1' + ',' + '$2');
    }

    var number = x1 + x2;
    number = number.replace(/0/g, '۰');
    number = number.replace(/1/g, '۱');
    number = number.replace(/2/g, '۲');
    number = number.replace(/3/g, '۳');
    number = number.replace(/4/g, '۴');
    number = number.replace(/5/g, '۵');
    number = number.replace(/6/g, '۶');
    number = number.replace(/7/g, '۷');
    number = number.replace(/8/g, '۸');
    number = number.replace(/9/g, '۹');

    return number;
}

function setPlansPrices() {
    var selectedDuration = $("input[name='mainPlanDuration']:checked").val();

    $.each(plansPrices, function(plan, price) {
        var totalAmount = price;
        var payableAmount = price;

        var durationTitle = '';

        if (selectedDuration) {
            switch (selectedDuration) {
                case 'monthly':
                    totalAmount = '';
                    durationTitle = 'ماهانه';
                    break
                case 'biannual':
                    totalAmount = totalAmount * 6;
                    payableAmount = totalAmount - Math.floor(totalAmount * 0.06);
                    durationTitle = '۶ ماهه';
                    break
                case 'annual':
                    totalAmount = totalAmount * 12;
                    payableAmount = payableAmount * 10;
                    durationTitle = 'سالانه';
                    break
            }
        }

        if (selectedDuration != 'monthly' && totalAmount == 0){
            $('#' + plan + 'PlanDiscountlessPriceView').text(numberFormat('-'));
        }else{
            $('#' + plan + 'PlanDiscountlessPriceView').text(numberFormat(totalAmount));
        }
        $('#' + plan + 'PlanPriceView').text(numberFormat(payableAmount));
        $('.mainPlanDurationView').text(durationTitle);
    });

    // set customization modal plan duration
    if (selectedDuration) {
        $("input[name='customizationPlanDuration']:checked").attr('checked', false);

        switch (selectedDuration) {
            case 'monthly':
                $("#customizationPlanDurationMonthly").parent().click();
                $("#customizationPlanDurationMonthly").attr('checked', true);
                break
            case 'biannual':
                $("#customizationPlanDurationBiannual").parent().click();
                $("#customizationPlanDurationBiannual").attr('checked', true);
                break
            case 'annual':
                $("#customizationPlanDurationAnnual").parent().click();
                $("#customizationPlanDurationAnnual").attr('checked', true);
                break
        }
    }
}

var updateCustomizationPrice= (function () {
    var updatePrice = function (data) {
        var learnerCount = $('#learnerCount').val();
        var extraAmount = (learnerCount - planBaseLearnerCount) * extraLearnerPrice;
        var monthlyAmount = planBasePrice + extraAmount;
        var totalAmount = monthlyAmount;
        var payableAmount = totalAmount;
        var durationTitle = '';

        var selectedDuration = $("input[name='customizationPlanDuration']:checked").val();

        if (selectedDuration) {
            switch (selectedDuration) {
                case 'monthly':
                    totalAmount = '';
                    durationTitle = 'ماهانه';
                    break
                case 'biannual':
                    totalAmount = totalAmount * 6;
                    payableAmount = totalAmount - Math.floor(totalAmount * 0.06);
                    durationTitle = '۶ ماهه';
                    break
                case 'annual':
                    totalAmount = totalAmount * 12;
                    payableAmount = payableAmount * 10;
                    durationTitle = 'سالانه';
                    break
            }
        }

        $('#planBasePriceView').text(numberFormat(planBasePrice));
        $('#learnerCountView').text(numberFormat(learnerCount));
        $('#extraAmountView').text(numberFormat(extraAmount));
        $('#monthlyAmountView').text(numberFormat(monthlyAmount));
        $('#totalAmountView').text(numberFormat(totalAmount));
        $('#payableAmountView').text(numberFormat(payableAmount));
        $('.customizationPlanDurationView').text(durationTitle);
    };
    return updatePrice;
})();


function setPlan(plan, planTitle) {
    var planDuration = 1;

    var selectedDuration = $("input[name='mainPlanDuration']:checked").val();

    if (selectedDuration) {
        switch (selectedDuration) {
            case 'monthly':
                planDuration = 1;
                break
            case 'biannual':
                planDuration = 6;
                break
            case 'annual':
                planDuration = 12;
                break
        }
    }

    registerFormSetPlan(plan, planTitle, planDuration);
}

function setPlanAndRedirectToRegister(url, email, plan, learnerCount) {
    var planDuration = 1;

    var selectedDuration = $("input[name='mainPlanDuration']:checked").val();

    if (selectedDuration) {
        switch (selectedDuration) {
            case 'monthly':
                planDuration = 1;
                break
            case 'biannual':
                planDuration = 6;
                break
            case 'annual':
                planDuration = 12;
                break
        }
    }

    var redirectUrl = url + '?email=' + email + '&plan=' + plan + '&duration=' + planDuration

    if (learnerCount) {
        redirectUrl += '&learnerCount=' + learnerCount
    }

    console.log(redirectUrl)

    window.location = redirectUrl
}

function customizationSetPlan(plan, planTitle) {
    var planDuration = 1;

    var selectedDuration = $("input[name='customizationPlanDuration']:checked").val();

    if (selectedDuration) {
        switch (selectedDuration) {
            case 'monthly':
                planDuration = 1;
                break
            case 'biannual':
                planDuration = 6;
                break
            case 'annual':
                planDuration = 12;
                break
        }
    }

    var learnerCount = $('#learnerCount').val();

    registerFormSetPlan(plan, planTitle, planDuration, learnerCount);
}

function customizationSetPlanAndRedirectToRegister(url, email, plan) {
    var planDuration = 1;

    var selectedDuration = $("input[name='customizationPlanDuration']:checked").val();

    if (selectedDuration) {
        switch (selectedDuration) {
            case 'monthly':
                planDuration = 1;
                break
            case 'biannual':
                planDuration = 6;
                break
            case 'annual':
                planDuration = 12;
                break
        }
    }

    var learnerCount = $('#learnerCount').val();

    setPlanAndRedirectToRegister(url, email, plan, learnerCount);
}

function updateRegisterPrice(){
    var monthlyBasePrice = $('#monthlyBasePrice').text();
    var basePrice
    var price = 0;
    var durationTitle = '';
    var duration = 1;

    var selectedDuration = $("input[name='registerPlanDuration']:checked").val();

    if (selectedDuration) {
        switch (selectedDuration) {
            case 'monthly':
                basePrice = '';
                price = monthlyBasePrice;
                durationTitle = 'ماهانه';
                duration = 1;
                break
            case 'biannual':
                basePrice = monthlyBasePrice * 6;
                price = basePrice - Math.floor(basePrice * 0.06);
                durationTitle = '۶ ماهه';
                duration = 6;
                break
            case 'annual':
                basePrice = monthlyBasePrice * 12;
                price = monthlyBasePrice * 10;
                durationTitle = 'سالانه';
                duration = 12;
                break
        }
    }

    $('#basePriceView').text(numberFormat(basePrice));
    $('#priceView').text(numberFormat(price));
    $('#durationView').text(numberFormat('(' + durationTitle + ')'));
    $('#duration').val(duration);
};
