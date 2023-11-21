<!doctype html>
<html lang="fa" dir="rtl">
	<head>
		<meta charset="UTF-8">
		<meta name="viewport" content="initial-scale=1.0,maximum-scale=1.0,user-scalable=no">
		<meta http-equiv="X-UA-Compatible" content="IE=edge">
		<meta name="msapplication-tap-highlight" content="no"/>
		<link rel="icon" type="image/png" href="{{ URL::asset('assets/img/favicon-16x16.png') }}" sizes="16x16">
		<link rel="icon" type="image/png" href="{{ URL::asset('assets/img/favicon-32x32.png') }}" sizes="32x32">
		<title>سیستم اطلاعاتی ثبت و نگهداری آثار تاریخی موزه</title>
		<link rel="stylesheet" href="{{ URL::asset('assets/css/uikit.rtl.css') }}" media="all">
		<link rel="stylesheet" href="{{ URL::asset('assets/icons/flags/flags.min.css') }}" media="all">
		<link rel="stylesheet" href="{{ URL::asset('assets/css/style_switcher.min.css') }}" media="all">
		<link rel="stylesheet" href="{{ URL::asset('assets/css/main.min.css') }}" media="all">
		<link rel="stylesheet" href="{{ URL::asset('assets/css/themes/themes_combined.min.css') }}" media="all">
		<link rel="stylesheet" href="{{ URL::asset('assets/css/custom.css') }}" media="all">
	</head>
	<body class=" sidebar_main_open sidebar_main_swipe">
		<!-- main header -->
		<header id="header_main">
			<div class="header_main_content">
				<nav class="uk-navbar">
					<!-- main sidebar switch -->
					<a href="list.html#" id="sidebar_main_toggle" class="sSwitch sSwitch_left">
					<span class="sSwitchIcon"></span>
					</a>
					<div class="uk-navbar-flip">
						<ul class="uk-navbar-nav user_actions">
							<li><a href="list.html#" id="full_screen_toggle" class="user_action_icon uk-visible-large"><i class="material-icons md-24 md-light">&#xE5D0;</i></a></li>
							<li data-uk-dropdown="{mode:'click',pos:'bottom-right'}">
								<a href="javascript::void();" class="user_action_image">{{ Auth::user()->name }} {{ Auth::user()->lname }} <img class="md-user-image" src="{{ URL::asset('assets/img/avatars/avatar_09.png') }}" alt=""/></a>
								<div class="uk-dropdown uk-dropdown-small">
									<ul class="uk-nav js-uk-prevent">
										<li><a href="{{ URL::to('users/edit') }}/{{ Auth::user()->id }}">پروفایل من</a></li>
										<li><a href="{{ URL::to('auth/logout') }}">خروج</a></li>
									</ul>
								</div>
							</li>
						</ul>
					</div>
				</nav>
			</div>
		</header>
		<!-- main header end -->
		<!-- main sidebar -->
		<aside id="sidebar_main">
			<div class="sidebar_main_header">
				<div class="sidebar_logo">
					<a href="{{ URL::to('/') }}" class="sSidebar_hide sidebar_logo_large" style="color: #ffffff; background-color: rgba(0 , 0 , 0 , 0.6); padding: 0 6px 3px 7px; font-size: 16px; line-height: 32px; margin-right: 0;">
						سیستم اطلاعاتی موزه
					</a>
				</div>
			</div>
			<div class="menu_section">
				<ul>
					{{-- <li title="داشبورد">
						<a href="{{ URL::to('/') }}">
						<span class="menu_icon"><i class="material-icons">&#xE871;</i></span>
						<span class="menu_title">داشبورد</span>
						</a>
					</li> --}}
					<li title="موزه ها">
						<a href="{{ URL::to('museums') }}">
						<span class="menu_icon"><i class="material-icons">account_balance</i></span>
						<span class="menu_title">موزه ها</span>
						</a>
					</li>
					<li title="اشیای تاریخی">
						<a href="{{ URL::to('items') }}">
						<span class="menu_icon"><i class="material-icons">filter_vintage</i></span>
						<span class="menu_title">اشیای تاریخی</span>
						</a>
					</li>
					<li title="دوره های تاریخی">
						<a href="{{ URL::to('historicalperiods') }}">
						<span class="menu_icon"><i class="material-icons">alarm</i></span>
						<span class="menu_title">دوره های تاریخی</span>
						</a>
					</li>
					<li title="کشور ها">
						<a href="{{ URL::to('countries') }}">
						<span class="menu_icon"><i class="material-icons">local_play</i></span>
						<span class="menu_title">کشور ها</span>
						</a>
					</li>
					<li title="شهرها">
						<a href="{{ URL::to('cities') }}">
						<span class="menu_icon"><i class="material-icons">location_city</i></span>
						<span class="menu_title">شهرها</span>
						</a>
					</li>
					<li title="کاربران">
						<a href="{{ URL::to('users') }}">
						<span class="menu_icon"><i class="material-icons">people</i></span>
						<span class="menu_title">کاربران</span>
						</a>
					</li>
					<li title="پروفایل من">
						<a href="{{ URL::to('users/edit') }}/{{ Auth::user()->id }}">
						<span class="menu_icon"><i class="material-icons">person</i></span>
						<span class="menu_title">پروفایل من</span>
						</a>
					</li>
					<li title="خروج">
						<a href="{{ URL::to('auth/logout') }}">
						<span class="menu_icon"><i class="material-icons">lock</i></span>
						<span class="menu_title">خروج</span>
						</a>
					</li>
				</ul>
			</div>
			<div style="text-align: center; width: 100%; position: absolute; bottom: 0; color: #424242; font-size: 11px; margin-bottom: 10px;">Designed by <a href="http://ahmadian.me" target="blank">Mehran Ahmadian</a></div>
		</aside>
		<!-- main sidebar end -->
		<div id="page_content">
			<div id="page_content_inner">
				<div id="top_bar">
					<ul id="breadcrumbs">
						@yield('breadcrumbs')
					</ul>
				</div>
				<div class="md-card uk-margin-medium-bottom">
					<div class="md-card-content">
						@include('common.errors')
						@yield('content')
					</div>
				</div>
			</div>
		</div>
		<div>
		</div>
		<script src="{{ URL::asset('assets/js/common.min.js') }}"></script>			
		<script src="{{ URL::asset('assets/js/uikit_custom.min.js') }}"></script>			
		<script src="{{ URL::asset('assets/js/admin_common.min.js') }}"></script>
		<script>
			$(function() {
				if(isHighDensity) {
					// enable hires images
					altair_helpers.retina_images();
				}
				if(Modernizr.touch) {
					// fastClick (touch devices)
					FastClick.attach(document.body);
				}
			});
			$window.load(function() {
				// ie fixes
				altair_helpers.ie_fix();
			});
		</script>
	</body>
</html>
