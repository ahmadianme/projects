@extends('layouts.app')

@section('breadcrumbs')
	<li><span>خانه</span></li>
	<li><span>دوره های تاریخی</span></li>
@endsection

@section('content')
	<form method="post">
		{{ csrf_field() }}
		<h3 class="heading_a">ثبت مورد جدید</h3>
		<hr style="margin: 20px 0 25px 0;">
		<div class="uk-grid" data-uk-grid-margin>
			<div class="uk-width-medium-1-1">
				<div class="uk-form-row md-card">
					<label>نام دوره</label>
					<input type="text" name="name" value="{{ old('name') }}{{ (isset($record) && !old('name')) ? $record->name : '' }}" class="md-input" />
				</div>
			</div>
			<div class="uk-width-medium-1-2">
				<div class="uk-form-row md-card">
					<label>سال شروع دوره</label>
					<input type="text" name="begin_year" value="{{ old('begin_year') }}{{ (isset($record) && !old('begin_year')) ? $record->begin_year : '' }}" class="md-input" />
				</div>
			</div>
			<div class="uk-width-medium-1-2">
				<div class="uk-form-row md-card">
					<label>سال پایان دوره</label>
					<input type="text" name="end_year" value="{{ old('end_year') }}{{ (isset($record) && !old('end_year')) ? $record->end_year : '' }}" class="md-input" />
				</div>
			</div>
		</div>
		<hr style="margin: 25px 0 15px 0">
		<div class="uk-grid">
			<div class="uk-width-1-7">
				<button type="submit" href="#" class="md-btn md-btn-success">ذخیره</button>
				<a href="{{ URL::to('historicalperiods') }}"><button type="button" class="md-btn md-btn-default" style="margin-right: 10px;">انصراف</button></a>
			</div>
		</div>
	<form>
@endsection