@extends('layouts.app')

@section('breadcrumbs')
	<li><span>خانه</span></li>
	<li><span>موزه ها</span></li>
@endsection

@section('content')
	<form method="post">
		{{ csrf_field() }}
		<h3 class="heading_a">اطلاعات موزه</h3>
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
	                	<select name="city_id" id="city_id" class="md-input" data-uk-tooltip="{pos:'top'}">
		                    <option value="" disabled="" selected="" hidden="">شهر</option>
		                    @foreach ($cities as $i => $city)
		                    	<option value="{{ $i }}" {{ (old('city_id') == $i || (isset($record) && !old('city_id') && $record->city_id == $i)) ? 'selected="selected"' : '' }}>{{ $city }}</option>
		                    @endforeach
	                	</select>
	                	<span class="md-input-bar "></span>
	                </div>
                </div>                            
            </div>
            <div class="uk-width-medium-1-2">
				<div class="uk-form-row md-card">
					<label>مساحت (متر مربع)</label>
					<input type="text" name="area" value="{{ old('area') }}{{ (isset($record) && !old('area')) ? $record->area : '' }}" class="md-input" />
				</div>
			</div>
			<div class="uk-width-medium-1-2">
				<div class="uk-form-row md-card">
					<label>تعداد سالن</label>
					<input type="text" name="num_of_halls" value="{{ old('num_of_halls') }}{{ (isset($record) && !old('num_of_halls')) ? $record->num_of_halls : '' }}" class="md-input" />
				</div>
			</div>
			<div class="uk-width-medium-1-2">
				<div class="uk-form-row md-card">
					<label>تلفن</label>
					<input type="text" name="phone" value="{{ old('phone') }}{{ (isset($record) && !old('phone')) ? $record->phone : '' }}" class="md-input" />
				</div>
			</div>
			<div class="uk-width-medium-1-2">
				<div class="uk-form-row md-card">
					<label>ایمیل</label>
					<input type="text" name="email" value="{{ old('email') }}{{ (isset($record) && !old('email')) ? $record->email : '' }}" class="md-input" />
				</div>
			</div>
			<div class="uk-width-medium-1-1">
				<div class="uk-form-row md-card">
					<label>آدرس</label>
					<input type="text" name="address" value="{{ old('address') }}{{ (isset($record) && !old('address')) ? $record->address : '' }}" class="md-input" />
				</div>
			</div>
		</div>
		<hr style="margin: 25px 0 15px 0">
		<div class="uk-grid">
			<div class="uk-width-1-7">
				<button type="submit" href="#" class="md-btn md-btn-success">ذخیره</button>
				<a href="{{ URL::to('museums') }}"><button type="button" class="md-btn md-btn-default" style="margin-right: 10px;">انصراف</button></a>
			</div>
		</div>
	<form>
@endsection