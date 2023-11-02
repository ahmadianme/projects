
<!doctype html>
<html lang="fa">

<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="keywords" content="NASH, AI, Artificial Intelligence">
    <meta name="description" content="NASH Diagnosis AI - Dayan AI Lab">

    <title>NASH Diagnosis AI</title>

    <style>
        @keyframes hideLoader{0%{ width: 100%; height: 100%; }100%{ width: 0; height: 0; }  }  body > div.loader{ position: fixed; background: white; width: 100%; height: 100%; z-index: 1071; opacity: 0; transition: opacity .5s ease; overflow: hidden; pointer-events: none; display: flex; align-items: center; justify-content: center;}body:not(.loaded) > div.loader{ opacity: 1;}body:not(.loaded){ overflow: hidden;}  body.loaded > div.loader{animation: hideLoader .5s linear .5s forwards;  } /* Typing Animation */.loading-animation {width: 6px;height: 6px;border-radius: 50%;animation: typing 1s linear infinite alternate;position: relative;left: -12px;}@keyframes typing {0% {background-color: rgba(100,100,100, 1);box-shadow: 12px 0px 0px 0px rgba(100,100,100, 0.2),24px 0px 0px 0px rgba(100,100,100, 0.2);}25% {background-color: rgba(100,100,100, 0.4);box-shadow: 12px 0px 0px 0px rgba(100,100,100, 2),24px 0px 0px 0px rgba(100,100,100, 0.2);}75% {background-color: rgba(100,100,100, 0.4);box-shadow: 12px 0px 0px 0px rgba(100,100,100, 0.2),24px 0px 0px 0px rgba(100,100,100, 1);}}
    </style>

    <script type="text/javascript">
        window.addEventListener("load", function () {    document.querySelector('body').classList.add('loaded');  });
    </script>

    <link rel="apple-touch-icon" sizes="180x180" href="assets/img/logo/logo.png">
    <link rel="icon" type="image/png" sizes="32x32" href="assets/img/logo/logo.png">
    <link rel="icon" type="image/png" sizes="16x16" href="assets/img/logo/logo.png">
    <link rel="shortcut icon" type="image/png" href="assets/img/logo/logo.png"/>
    <meta name="msapplication-TileColor" content="#da532c">
    <meta name="theme-color" content="#ffffff">

    <link href="assets/css/style.css" rel="stylesheet" type="text/css" media="all" />
    <link href="assets/css/colors.css" rel="stylesheet" type="text/css" media="all" />
    <link href="assets/css/theme.css" rel="stylesheet" type="text/css" media="all" />
    <link href="assets/css/rtl.css" rel="stylesheet" type="text/css" media="all" />
    <link rel="preload" as="font" href="assets/fonts/persian-fn.woff2" type="font/woff2" crossorigin="anonymous">

    <script type="text/javascript" src="assets/js/jquery.min.js"></script>
    <script type="text/javascript" src="assets/js/popper.min.js"></script>
    <script type="text/javascript" src="assets/js/bootstrap.js"></script>

    <script type="text/javascript" src="assets/js/flickity.pkgd.min.js"></script>


    <link href="assets/vendor/star-rating/star-rating.min.css" media="all" rel="stylesheet" type="text/css" />
    <link href="assets/vendor/star-rating/theme.css" media="all" rel="stylesheet" type="text/css" />
    <script src="assets/vendor/star-rating/star-rating.min.js" type="text/javascript"></script>
    <script src="assets/vendor/star-rating/theme.js"></script>

    <script>
        function showLoading() {
            $('#loading').removeClass('hidden')
        }

        function hideLoading() {
            $('#loading').addClass('hidden')
        }

        function showMessage(type, title, text) {
            $('#message-modal .modal-content').removeClass('bg-success')
            $('#message-modal .modal-content').removeClass('bg-danger')
            $('#message-modal .modal-content').removeClass('bg-warning')
            $('#message-modal .modal-content').removeClass('bg-info')
            $('#message-modal .modal-content').removeClass('bg-primary')
            $('#message-modal .modal-content').removeClass('bg-light')
            $('#message-modal .modal-content').removeClass('bg-white')

            $('#message-modal .modal-content').addClass('bg-' + type)

            // if (type == 'light' || type == '')

            $('#message-modal #message-modal-title').html(title)
            $('#message-modal #message-modal-text').html(text)

            $('#message-modal').modal()
        }

        function submitQuery (query) {
            if (query) {
                $('#query').val(query);
            }
            $('#summerizerForm').submit()

            return false;
        }

        $(document).ready(function () {
            $('#sample-loader').change(function (){
                fillSample(this.value)
            })

            $('#inference-form-submit-button').click(function (){
                const formFields = [
                    "Age",
                    "Gender (female)",
                    "Nationality",
                    "BMI",
                    "History.of.hbv.vaccine",
                    "Diabetes",
                    "Hypertension",
                    "Fatty.Liver",
                    "High.blood.fats",
                    "FBS",
                    "TG",
                    "Chol",
                    "HDL",
                    "LDL",
                    "SGOT",
                    "SGPT",
                    "Alk",
                    "GGT",
                    "Alb.s",
                    "HCVAb",
                    "Thyroid_activity",
                ]

                let inputs = {}
                let error = false

                formFields.forEach(function(key){
                    var value = $('#inference-form input[name="'+key+'"]').val()

                    if (!value || value == '') {
                        showMessage('danger', 'Error', 'All fields are required.')
                        error = true
                    }

                    inputs[key] = [value]
                })

                if (!error) {
                    const data = {
                        'inputs': inputs
                    };

                    console.log(data)


                    $.ajax({
                        type: "POST",
                        url: "https://ai.dayansystem.com/api/nash/predict",
                        data: JSON.stringify(data),
                        contentType: "application/json; charset=utf-8",
                        dataType: "json",
                        beforeSend: function(){
                            showLoading()
                        },
                        dataType: "json",
                        success: function(response){
                            let text = ''

                            Object.keys(response.modelOutput.labelProbabilities[0]).forEach(function(label){
                                console.log(label)
                                console.log(response.modelOutput.labelProbabilities[0][label])
                                text += label + ': ' + response.modelOutput.labelProbabilities[0][label] + '<br>'
                            })

                            showMessage('success', response.modelOutput.labels[0], text)
                            hideLoading()
                        },
                        error: function(errMsg) {
                            console.log(errMsg)
                            showMessage('danger', 'Error', 'There was an error proccessing your request.')
                            hideLoading()
                        }
                    });
                }
            })
        })

        function fillSample(sampleId) {
            const samples = {
                1: {
                    "Age": 40.0,
                    "Gender (female)": 0,
                    "Nationality": 1.0,
                    "BMI": 21.55102040816326,
                    "History.of.hbv.vaccine": 0.1223021582733813,
                    "Diabetes": 0.0,
                    "Hypertension": 0.0,
                    "Fatty.Liver": 0.0,
                    "High.blood.fats": 0.0,
                    "FBS": 98.0,
                    "TG": 43.0,
                    "Chol": 256.0,
                    "HDL": 47.0,
                    "LDL": 165.0,
                    "SGOT": 0.5,
                    "SGPT": 0.2,
                    "Alk": 16.9,
                    "GGT": 51.1,
                    "Alb.s": 87.0,
                    "HCVAb": 28.6,
                    "Thyroid_activity": 3.5,
                },
                2: {
                    "Age": 34.0,
                     "Gender (female)": 0,
                     "Nationality": 0.0,
                     "BMI": 34.95521363253332,
                     "History.of.hbv.vaccine": 1.0,
                     "Diabetes": 0.0,
                     "Hypertension": 0.0,
                     "Fatty.Liver": 0.0,
                     "High.blood.fats": 0.0,
                     "FBS": 315.0,
                     "TG": 34.0,
                     "Chol": 212.0,
                     "HDL": 41.0,
                     "LDL": 140.0,
                     "SGOT": 0.6,
                     "SGPT": 0.2,
                     "Alk": 17.2,
                     "GGT": 51.7,
                     "Alb.s": 88.0,
                     "HCVAb": 29.2,
                     "Thyroid_activity": 4.3,
                },
                3: {
                    "Age": 39.0,
                    "Gender (female)": 0,
                    "Nationality": 0.0,
                    "BMI": 27.16691927135463,
                    "History.of.hbv.vaccine": 0.1223021582733813,
                    "Diabetes": 0.0,
                    "Hypertension": 0.0,
                    "Fatty.Liver": 1.0,
                    "High.blood.fats": 0.0,
                    "FBS": 152.0,
                    "TG": 38.0,
                    "Chol": 235.0,
                    "HDL": 54.0,
                    "LDL": 146.0,
                    "SGOT": 0.3,
                    "SGPT": 0.1,
                    "Alk": 16.1,
                    "GGT": 50.1,
                    "Alb.s": 84.0,
                    "HCVAb": 26.8,
                    "Thyroid_activity": 3.7,
                }
            }

            const sample = samples[sampleId]

            Object.keys(sample).forEach(function (key){
                $('#inference-form input[name="'+key+'"]').val(sample[key])
            })
        }
    </script>

    <style>
        .hidden{display: none!important;}
        .lds-roller{display:inline-block;position:relative;width:80px;height:80px}.lds-roller div{animation:1.2s cubic-bezier(.5,0,.5,1) infinite lds-roller;transform-origin:40px 40px}.lds-roller div:after{content:" ";display:block;position:absolute;width:7px;height:7px;border-radius:50%;background:#fff;margin:-4px 0 0 -4px}.lds-roller div:first-child{animation-delay:-36ms}.lds-roller div:first-child:after{top:63px;left:63px}.lds-roller div:nth-child(2){animation-delay:-72ms}.lds-roller div:nth-child(2):after{top:68px;left:56px}.lds-roller div:nth-child(3){animation-delay:-108ms}.lds-roller div:nth-child(3):after{top:71px;left:48px}.lds-roller div:nth-child(4){animation-delay:-144ms}.lds-roller div:nth-child(4):after{top:72px;left:40px}.lds-roller div:nth-child(5){animation-delay:-.18s}.lds-roller div:nth-child(5):after{top:71px;left:32px}.lds-roller div:nth-child(6){animation-delay:-216ms}.lds-roller div:nth-child(6):after{top:68px;left:24px}.lds-roller div:nth-child(7){animation-delay:-252ms}.lds-roller div:nth-child(7):after{top:63px;left:17px}.lds-roller div:nth-child(8){animation-delay:-288ms}.lds-roller div:nth-child(8):after{top:56px;left:12px}@keyframes lds-roller{0%{transform:rotate(0)}100%{transform:rotate(360deg)}}

        .loading-container{
            position: fixed;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.6);
            z-index: 10000;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .rating-input{
            position: absolute !important;
            right: 125px;
            left: auto;
            /*display: inline-block;*/
            width: auto;
        }

        .inference-form input{
            color: #fff;
            font-weight: bold;
            min-height: 48px
        }

        .inference-form input:focus{
            color: #fff;
        }

        .inference-form .input-group .input-group-text{
            /*color: #212529;*/
            /*background-color: #ffc107;*/
            /*border-color: #ffc107;*/
            min-width: 160px;
            justify-content: center;
        }

        .input-group>.custom-select:not(:last-child), .input-group>.form-control:not(:last-child) {
            border-top-right-radius: 6px;
            border-bottom-right-radius: 6px;
        }

        #message-modal .close svg *{
            fill: #fff;
        }
    </style>
</head>

<body>

<div id="loading" class="loading-container hidden">
    <div class="lds-roller"><div></div><div></div><div></div><div></div><div></div><div></div><div></div><div></div></div>
</div>

<div class="navbar-container">
    <nav class="navbar navbar-expand-lg justify-content-between navbar-light" data-sticky="top">
        <div class="container">
            <div class="flex-fill px-0 d-flex justify-content-between">
                <button class="navbar-toggler" type="button" data-toggle="collapse" data-target=".navbar-collapse" aria-expanded="false" aria-label="Toggle navigation">
                    <img class="icon navbar-toggler-open" src="assets/img/icons/interface/menu.svg" alt="menu interface icon" data-inject-svg />
                    <img class="icon navbar-toggler-close" src="assets/img/icons/interface/cross.svg" alt="cross interface icon" data-inject-svg />
                </button>

                <a class="navbar-brand mr-0 pr-7 p-lg-0 d-inline-block" href="#inference-form">
                    <img src="assets/img/dayam-ai-lab-dark-sm.png" class="logo-dark" alt="logo">
                    <img src="assets/img/dayam-ai-lab-light-sm.png" class="logo-light" alt="logo">
                </a>
            </div>

            <div class="collapse navbar-collapse px-0 px-lg-2 flex-fill flex justify-content-center">
                <div class="py-2 py-lg-0">
                    <ul class="navbar-nav pr-3">

                        <li class="nav-item">
                            <a href="#inference-form" data-smooth-scroll class="nav-link">Home</a>
                        </li>
                    </ul>
                </div>
            </div>

            <div class="flex-fill px-0 d-flex justify-content-end">
                <a class="navbar-brand mr-0 pr-7 p-lg-0 d-inline-block" href="">
                    <img src="assets/img/universities-sm.png" class="logo-dark" alt="logo">
                    <img src="assets/img/universities-sm.png" class="logo-light" alt="logo">
                </a>
            </div>
        </div>
    </nav>
</div>

<div class="modal fade" id="message-modal" tabindex="-1" role="dialog"
     aria-hidden="true">
    <div class="modal-dialog text-light" role="document">
        <div class="modal-content border-0">
            <div class="modal-body">
                <button type="button" class="close" data-dismiss="modal" aria-label="Close">
                    <img class="icon icon-md bg-white" src="assets/img/icons/interface/cross.svg" alt="cross interface icon" data-inject-svg/>
                </button>
                <div class="m-3 d-flex align-items-center">
                    <div class="ml-3">
                        <h5 id="message-modal-title" class="mb-1"></h5>
                        <span id="message-modal-text"></span>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<div id="inference-form" class="bg-gradient o-hidden position-relative pt-5" data-overlay>
    <section class="has-divider text-light">
        <div class="container layer-2 pb-0 pt-6">
            <div class="row justify-content-center">
                <div class="col-12 text-center text-lg-left mb-5 mb-lg-0">
                    <p class="display-9 font-weight-bold">Non-Alcoholic SteatoHepatitis (NASH) Diagnosis AI</p>

                    <div class="my-2">
                        <span class="h4 display-12 text-warning" data-typed-text data-loop="true" data-type-speed="40" data-strings='[
                            "AI based NASH diagnosis system...",
                            "Neural Network based inference...",
                            "Fast inference under 5 ms..."
                        ]'></span>
                    </div>

                    <div class="row mt-5">


                        <div class="col-12 inference-form">
                            <form id="summerizerForm" action="" method="post" class="mt-2 d-flex flex-column form-group">
                                <input type="hidden" name="inference-form-submitted" value="1" />

                                <div class="row">
                                    <div class="col-12 mb-4">
                                        <div class="input-group">
                                            <div class="input-group-prepend"><span class="input-group-text">Load Test Samples</span></div>
                                            <select class="custom-select bg-light" id="sample-loader">
                                                <option selected></option>
                                                <option value="1">Sample 1 (LE Homogene)</option>
                                                <option value="2">Sample 2 (LE Grade 1)</option>
                                                <option value="3">Sample 3 (LE Grade 2)</option>
                                            </select>
                                            <img class="icon" src="assets/img/icons/interface/arrow-caret.svg" alt="arrow-caret interface icon" data-inject-svg />
                                        </div>
                                    </div>

                                    <div class="col-12 col-md-6 col-lg-4 mb-2">
                                        <div class="input-group">
                                            <div class="input-group-prepend"><span class="input-group-text bg-light text-dark border-0">Age</span></div>
                                            <input class="form-control bg-transparent" name="Age" required />
                                        </div>
                                    </div>

                                    <div class="col-12 col-md-6 col-lg-4 mb-2">
                                        <div class="input-group">
                                            <div class="input-group-prepend"><span class="input-group-text bg-light text-dark border-0">Gender (female)</span></div>
                                            <input class="form-control bg-transparent" name="Gender (female)" required />
                                        </div>
                                    </div>

                                    <div class="col-12 col-md-6 col-lg-4 mb-2">
                                        <div class="input-group">
                                            <div class="input-group-prepend"><span class="input-group-text bg-light text-dark border-0">Nationality</span></div>
                                            <input class="form-control bg-transparent" name="Nationality" required />
                                        </div>
                                    </div>

                                    <div class="col-12 col-md-6 col-lg-4 mb-2">
                                        <div class="input-group">
                                            <div class="input-group-prepend"><span class="input-group-text bg-light text-dark border-0">BMI</span></div>
                                            <input class="form-control bg-transparent" name="BMI" required />
                                        </div>
                                    </div>

                                    <div class="col-12 col-md-6 col-lg-4 mb-2">
                                        <div class="input-group">
                                            <div class="input-group-prepend"><span class="input-group-text bg-light text-dark border-0">HBV Vaccine</span></div>
                                            <input class="form-control bg-transparent" name="History.of.hbv.vaccine" required />
                                        </div>
                                    </div>

                                    <div class="col-12 col-md-6 col-lg-4 mb-2">
                                        <div class="input-group">
                                            <div class="input-group-prepend"><span class="input-group-text bg-light text-dark border-0">Diabetes</span></div>
                                            <input class="form-control bg-transparent" name="Diabetes" required />
                                        </div>
                                    </div>

                                    <div class="col-12 col-md-6 col-lg-4 mb-2">
                                        <div class="input-group">
                                            <div class="input-group-prepend"><span class="input-group-text bg-light text-dark border-0">Hypertension</span></div>
                                            <input class="form-control bg-transparent" name="Hypertension" required />
                                        </div>
                                    </div>

                                    <div class="col-12 col-md-6 col-lg-4 mb-2">
                                        <div class="input-group">
                                            <div class="input-group-prepend"><span class="input-group-text bg-light text-dark border-0">Fatty Liver</span></div>
                                            <input class="form-control bg-transparent" name="Fatty.Liver" required />
                                        </div>
                                    </div>

                                    <div class="col-12 col-md-6 col-lg-4 mb-2">
                                        <div class="input-group">
                                            <div class="input-group-prepend"><span class="input-group-text bg-light text-dark border-0">High Blood Fats</span></div>
                                            <input class="form-control bg-transparent" name="High.blood.fats" required />
                                        </div>
                                    </div>

                                    <div class="col-12 col-md-6 col-lg-4 mb-2">
                                        <div class="input-group">
                                            <div class="input-group-prepend"><span class="input-group-text bg-light text-dark border-0">FBS</span></div>
                                            <input class="form-control bg-transparent" name="FBS" required />
                                        </div>
                                    </div>

                                    <div class="col-12 col-md-6 col-lg-4 mb-2">
                                        <div class="input-group">
                                            <div class="input-group-prepend"><span class="input-group-text bg-light text-dark border-0">TG</span></div>
                                            <input class="form-control bg-transparent" name="TG" required />
                                        </div>
                                    </div>

                                    <div class="col-12 col-md-6 col-lg-4 mb-2">
                                        <div class="input-group">
                                            <div class="input-group-prepend"><span class="input-group-text bg-light text-dark border-0">Chol</span></div>
                                            <input class="form-control bg-transparent" name="Chol" required />
                                        </div>
                                    </div>

                                    <div class="col-12 col-md-6 col-lg-4 mb-2">
                                        <div class="input-group">
                                            <div class="input-group-prepend"><span class="input-group-text bg-light text-dark border-0">HDL</span></div>
                                            <input class="form-control bg-transparent" name="HDL" required />
                                        </div>
                                    </div>

                                    <div class="col-12 col-md-6 col-lg-4 mb-2">
                                        <div class="input-group">
                                            <div class="input-group-prepend"><span class="input-group-text bg-light text-dark border-0">LDL</span></div>
                                            <input class="form-control bg-transparent" name="LDL" required />
                                        </div>
                                    </div>

                                    <div class="col-12 col-md-6 col-lg-4 mb-2">
                                        <div class="input-group">
                                            <div class="input-group-prepend"><span class="input-group-text bg-light text-dark border-0">SGOT</span></div>
                                            <input class="form-control bg-transparent" name="SGOT" required />
                                        </div>
                                    </div>

                                    <div class="col-12 col-md-6 col-lg-4 mb-2">
                                        <div class="input-group">
                                            <div class="input-group-prepend"><span class="input-group-text bg-light text-dark border-0">SGPT</span></div>
                                            <input class="form-control bg-transparent" name="SGPT" required />
                                        </div>
                                    </div>

                                    <div class="col-12 col-md-6 col-lg-4 mb-2">
                                        <div class="input-group">
                                            <div class="input-group-prepend"><span class="input-group-text bg-light text-dark border-0">Alk</span></div>
                                            <input class="form-control bg-transparent" name="Alk" required />
                                        </div>
                                    </div>

                                    <div class="col-12 col-md-6 col-lg-4 mb-2">
                                        <div class="input-group">
                                            <div class="input-group-prepend"><span class="input-group-text bg-light text-dark border-0">GGT</span></div>
                                            <input class="form-control bg-transparent" name="GGT" required />
                                        </div>
                                    </div>

                                    <div class="col-12 col-md-6 col-lg-4 mb-2">
                                        <div class="input-group">
                                            <div class="input-group-prepend"><span class="input-group-text bg-light text-dark border-0">Alb.s</span></div>
                                            <input class="form-control bg-transparent" name="Alb.s" required />
                                        </div>
                                    </div>

                                    <div class="col-12 col-md-6 col-lg-4 mb-2">
                                        <div class="input-group">
                                            <div class="input-group-prepend"><span class="input-group-text bg-light text-dark border-0">HCVAb</span></div>
                                            <input class="form-control bg-transparent" name="HCVAb" required />
                                        </div>
                                    </div>

                                    <div class="col-12 col-md-6 col-lg-4 mb-2">
                                        <div class="input-group">
                                            <div class="input-group-prepend"><span class="input-group-text bg-light text-dark border-0">Thyroid Activity</span></div>
                                            <input class="form-control bg-transparent" name="Thyroid_activity" required />
                                        </div>
                                    </div>

                                    <div class="col-12">
                                        <div class="d-flex justify-content-center my-2">
                                            <button class="btn btn-md btn-warning mr-2" type="button" id="inference-form-submit-button">Submit and Predict</button>
                                            <input type="reset" class="btn btn-md btn-outline-white" value="Clear Form"/>
                                        </div>
                                    </div>
                                </div>
                            </form>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <div class="decoration-wrapper d-none d-sm-block">
            <div data-jarallax-element="0 50">
                <div class="decoration middle-y right">
                    <img class="bg-white opacity-10 scale-4" src="assets/img/decorations/blob-3.svg" alt="Blob" data-inject-svg />
                </div>
            </div>
        </div>
<!--        <div class="divider">-->
<!--            <img class="bg-light white" src="assets/img/dividers/divider-2.svg" alt="جداساز" data-inject-svg />-->
<!--        </div>-->
    </section>
</div>




<footer class="py-3 bg-primary-3 text-light" id="footer">
    <div class="container">
        <div class="row justify-content-center">
            <div class="col col-md-auto text-center text-muted">
                    <p class="m-0">Copyright © 2023. All rights reserved.</p>
            </div>
        </div>
    </div>
</footer>

<a href="#" class="btn back-to-top btn-primary btn-round" data-smooth-scroll data-aos="fade-up" data-aos-offset="2000" data-aos-mirror="true" data-aos-once="false">
    <img class="icon" src="assets/img/icons/theme/navigation/arrow-up.svg" alt="arrow-up icon" data-inject-svg />
</a>

<script type="text/javascript" src="assets/js/magic-scroll/plugins/gsap.min.js"></script>
<script type="text/javascript" src="assets/js/magic-scroll/ScrollMagic.min.js"></script>
<script type="text/javascript" src="assets/js/magic-scroll/plugins/animation.gsap.min.js"></script>

<!-- AOS (Animate On Scroll - animates elements into view while scrolling down) -->
<script type="text/javascript" src="assets/js/aos.js"></script>
<!-- Clipboard (copies content from browser into OS clipboard) -->
<script type="text/javascript" src="assets/js/clipboard.js"></script>
<!-- Fancybox (handles image and video lightbox and galleries) -->
<!--<script type="text/javascript" src="assets/js/jquery.fancybox.min.js"></script>-->
<script type="text/javascript" src="assets/js/jquery.fancybox.js"></script>
<!-- Flatpickr (calendar/date/time picker UI) -->
<script type="text/javascript" src="assets/js/flatpickr.min.js"></script>

<!-- Ion rangeSlider (flexible and pretty range slider elements) -->
<script type="text/javascript" src="assets/js/ion.rangeSlider.min.js"></script>
<!-- Isotope (masonry layouts and filtering) -->
<script type="text/javascript" src="assets/js/isotope.pkgd.min.js"></script>
<!-- jarallax (parallax effect and video backgrounds) -->
<script type="text/javascript" src="assets/js/jarallax.min.js"></script>
<script type="text/javascript" src="assets/js/jarallax-video.min.js"></script>
<script type="text/javascript" src="assets/js/jarallax-element.min.js"></script>
<!-- jQuery Countdown (displays countdown text to a specified date) -->
<script type="text/javascript" src="assets/js/jquery.countdown.min.js"></script>
<!-- jQuery smartWizard facilitates steppable wizard content -->
<script type="text/javascript" src="assets/js/jquery.smartWizard.min.js"></script>
<!-- Plyr (unified player for Video, Audio, Vimeo and Youtube) -->
<script type="text/javascript" src="assets/js/plyr.polyfilled.min.js"></script>
<!-- Prism (displays formatted code boxes) -->
<script type="text/javascript" src="assets/js/prism.js"></script>
<!-- ScrollMonitor (manages events for elements scrolling in and out of view) -->
<script type="text/javascript" src="assets/js/scrollMonitor.js"></script>
<!-- Smooth scroll (animation to links in-page)-->
<script type="text/javascript" src="assets/js/smooth-scroll.polyfills.min.js"></script>
<!-- SVGInjector (replaces img tags with SVG code to allow easy inclusion of SVGs with the benefit of inheriting colors and styles)-->
<script type="text/javascript" src="assets/js/svg-injector.umd.production.js"></script>
<!-- TwitterFetcher (displays a feed of tweets from a specified account)-->
<script type="text/javascript" src="assets/js/twitterFetcher_min.js"></script>
<!-- Typed text (animated typing effect)-->
<script type="text/javascript" src="assets/js/typed.min.js"></script>
<!-- Required theme scripts (Do not remove) -->
<script type="text/javascript" src="assets/js/theme.js"></script>

<script type="text/javascript" src="assets/js/custom.js"></script>

</body>

</html>
