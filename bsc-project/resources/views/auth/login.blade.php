<!doctype html>
	<html lang="en" dir="rtl">        
		<head>
			<meta charset="UTF-8">
			<meta name="viewport" content="initial-scale=1.0,maximum-scale=1.0,user-scalable=no">
			<meta http-equiv="X-UA-Compatible" content="IE=edge">
			<meta name="msapplication-tap-highlight" content="no"/>

			<link rel="icon" type="image/png" href="assets/img/favicon-16x16.png" sizes="16x16">
			<link rel="icon" type="image/png" href="assets/img/favicon-32x32.png" sizes="32x32">
			<title>سیستم اطلاعاتی ثبت و نگهداری آثار تاریخی موزه</title>

			<link rel="stylesheet" href="{{ URL::asset('assets/css/uikit.rtl.css') }}" media="all">			
			<link rel="stylesheet" href="{{ URL::asset('assets/css/login_page.min.css') }}" />
			<link rel="stylesheet" href="{{ URL::asset('assets/css/custom.css') }}" media="all">
		</head>

		<body class="login_page" style="background: url({{ URL::asset('assets/img/background/login.jpg') }}) 0 50% no-repeat; background-size: 100% auto;">
			<div class="login_page_wrapper">
				<div class="md-card" id="login_card">
					<div style="padding: 20px; background: #007eff;">
						<h3 style="text-align: center; color: #fff; margin-bottom: 0;">سیستم اطلاعاتی موزه</h3>
					</div>
					@include('common.errors')
					<div class="md-card-content large-padding" id="login_form">
						<div class="login_heading">
							<div class="user_avatar"></div>
						</div>
						<form method="post">
							{{ csrf_field() }}
							<div class="uk-form-row">
								<label for="email">آدرس ایمیل</label>
								<input class="md-input" type="text" id="email" name="email" value="{{ old('email') }}" />
							</div>
							<div class="uk-form-row">
								<label for="password">گذرواژه</label>
								<input class="md-input" type="password" id="password" name="password" />
							</div>
							<div class="uk-margin-medium-top">
								<input type="submit" value="ورود" class="md-btn md-btn-primary md-btn-block md-btn-large">
							</div>
							<div style="text-align: center; width: 73%; position: absolute; bottom: 0; color: #424242; font-size: 11px; margin-bottom: 10px;">Designed by <a href="http://ahmadian.me" target="blank">Mehran Ahmadian</a></div>
						</form>
					</div>
				</div>
			</div>

			<script src="{{ URL::asset('assets/js/common.min.js') }}"></script>			
			<script src="{{ URL::asset('assets/js/uikit_custom.min.js') }}"></script>			
			<script src="{{ URL::asset('assets/js/admin_common.min.js') }}"></script>			
			<script src="{{ URL::asset('assets/js/pages/login.min.js') }}"></script>
		</body>
	</html>
