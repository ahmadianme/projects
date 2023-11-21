@extends('layouts.app')

@section('breadcrumbs')
	<li><span>خانه</span></li>
	<li><span>کشورها</span></li>
@endsection

@section('content')
	<form method="post">
		{{ csrf_field() }}
		<h3 class="heading_a">اطلاعات کشور</h3>
		<hr style="margin: 20px 0 25px 0;">
		<div class="uk-grid" data-uk-grid-margin>
			<div class="uk-width-medium-1-2">
				<div class="uk-form-row md-card">
					<label>نام</label>
					<input type="text" name="name" value="{{ old('name') }}{{ (isset($record) && !old('name')) ? $record->name : '' }}" class="md-input" />
				</div>
			</div>
			<div class="uk-width-medium-1-2">
            	<div class="md-card">
	                <div class="md-input-wrapper md-input-filled">
	                	<select name="continent" id="continent" class="md-input" data-uk-tooltip="{pos:'top'}">
		                    <option value="" disabled="" selected="" hidden="">قاره</option>
		                    <option value="asia" {{ (old('continent') == 'asia' || (isset($record) && !old('begin_year') && $record->continent == 'asia')) ? 'selected="selected"' : '' }}>آسیا</option>
		                    <option value="america" {{ (old('continent') == 'america' || (isset($record) && !old('begin_year') && $record->continent == 'america')) ? 'selected="selected"' : '' }}>آمریکا</option>
		                    <option value="europe" {{ (old('continent') == 'europe' || (isset($record) && !old('begin_year') && $record->continent == 'europe')) ? 'selected="selected"' : '' }}>اروپا</option>
		                    <option value="africa" {{ (old('continent') == 'africa' || (isset($record) && !old('begin_year') && $record->continent == 'africa')) ? 'selected="selected"' : '' }}>آفریقا</option>
		                    <option value="australia" {{ (old('continent') == 'australia' || (isset($record) && !old('begin_year') && $record->continent == 'australia')) ? 'selected="selected"' : '' }}>استرالیا</option>
	                	</select>
	                	<span class="md-input-bar "></span>
	                </div>
                </div>                            
            </div>
			<div class="uk-width-medium-1-2">
				<div class="uk-form-row md-card">
					<label>زبان</label>
					<input type="text" name="language" value="{{ old('language') }}{{ (isset($record) && !old('language')) ? $record->language : '' }}" class="md-input" />
				</div>
			</div>
			<div class="uk-width-medium-1-2">
				<div class="uk-form-row md-card">
					<label>منطقه زمانی</label>
					<input type="text" name="timezone" value="{{ old('timezone') }}{{ (isset($record) && !old('timezone')) ? $record->timezone : '' }}" class="md-input" />
				</div>
			</div>
		</div>
		<hr style="margin: 25px 0 15px 0">
		<div class="uk-grid">
			<div class="uk-width-1-7">
				<button type="submit" href="#" class="md-btn md-btn-success">ذخیره</button>
				<a href="{{ URL::to('countries') }}"><button type="button" class="md-btn md-btn-default" style="margin-right: 10px;">انصراف</button></a>
			</div>
		</div>
	<form>
@endsection