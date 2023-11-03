$(document).ready(function () {
    var carousel = $('.faq-carousel');
    var faqs = carousel.data('faqs');

    carousel.on('change.flickity', function (event, index) {
        console.log(index)

        if (faqs && faqs.hasOwnProperty(index)) {
            $('#faqTitle').html(faqs[index].c);

            if (faqs[index].hasOwnProperty('qs') && faqs[index].qs.length > 0) {
                var faqQuestions = '';

                $.each(faqs[index].qs, function( index, question ) {
                    faqQuestions += '<div class="card mb-2 card-sm card-body hover-shadow-sm" data-aos="fade-up" data-aos-delay="'+ ((index + 1) * 50) +'"><div data-target="#faq-panel-'+ index +'" class="accordion-panel-title" data-toggle="collapse" role="button"aria-expanded="false" aria-controls="faq-panel-'+ index +'"><span class="h6 mb-0">'+ question.q +'</span><img class="icon" src="/themes/insurance24.ir/assets/img/icons/interface/plus.svg" alt="plus interface icon"/></div><div class="collapse" id="faq-panel-'+ index +'"><div class="pt-3"><p class="mb-0">'+ question.a +'</p></div></div></div>';
                });

                $('#faqQuestions').html(faqQuestions);
            }else{
                $('#faqQuestions').html('<p class="text-center">پرسش و پاسخی در این بخش یافت نشد.</p>');
            }
        }else {
            $('#faqTitle').html('');
            $('#faqQuestions').html('<p class="text-center">پرسش و پاسخی در این بخش یافت نشد.</p>');
        }
    });
});
