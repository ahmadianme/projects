$(document).ready(function () {
    $(".scroll-sync-x").on("scroll", function() {
        $(".scroll-sync-x").scrollLeft($(this).scrollLeft());
    });

    $(".scroll-sync-y").on("scroll", function() {
        $(".scroll-sync-y").scrollTop($(this).scrollTop());
    });
});