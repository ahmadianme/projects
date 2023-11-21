@extends('layouts.app')

@section('breadcrumbs')
	<li><span>خانه</span></li>
	<li><span>شهرها</span></li>
@endsection

@section('content')
	<form method="post">
		{{ csrf_field() }}
		<h3 class="heading_a">اطلاعات شهر</h3>
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
	                	<select name="country_id" id="country_id" class="md-input" data-uk-tooltip="{pos:'top'}">
		                    <option value="" disabled="" selected="" hidden="">کشور</option>
		                    @foreach ($countries as $i => $country)
		                    	<option value="{{ $i }}" {{ (old('country_id') == $i || (isset($record) && !old('country_id') && $record->country_id == $i)) ? 'selected="selected"' : '' }}>{{ $country }}</option>
		                    @endforeach
	                	</select>
	                	<span class="md-input-bar "></span>
	                </div>
                </div>                            
            </div>
		</div>
		<hr style="margin: 25px 0 15px 0">
		<div class="uk-grid">
			<div class="uk-width-1-7">
				<button type="submit" href="#" class="md-btn md-btn-success">ذخیره</button>
				<a href="{{ URL::to('cities') }}"><button type="button" class="md-btn md-btn-default" style="margin-right: 10px;">انصراف</button></a>
			</div>
		</div>
	<form>
@endsection