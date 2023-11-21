@extends('layouts.app')

@section('breadcrumbs')
	<li><span>خانه</span></li>
	<li><span>اشیا تاریخی</span></li>
@endsection

@section('content')
	<form method="post" enctype="multipart/form-data">
		{{ csrf_field() }}
		<h3 class="heading_a">اطلاعات شیء</h3>
		<hr style="margin: 20px 0 25px 0;">
		<div class="uk-grid" data-uk-grid-margin>
			<div class="uk-width-medium-1-1">
				<div class="uk-form-row md-card">
					<label>نام</label>
					<input type="text" name="name" value="{{ old('name') }}{{ (isset($record) && !old('name')) ? $record->name : '' }}" class="md-input" />
				</div>
			</div>
			<div class="uk-width-medium-1-2">
            	<div class="md-card">
	                <div class="md-input-wrapper md-input-filled">
	                	<select name="museum_id" id="museum_id" class="md-input" data-uk-tooltip="{pos:'top'}">
		                    <option value="" disabled="" selected="" hidden="">موزه</option>
		                    @foreach ($museums as $i => $museum)
		                    	<option value="{{ $i }}" {{ (old('museum_id') == $i || (isset($record) && !old('museum_id') && $record->museum_id == $i)) ? 'selected="selected"' : '' }}>{{ $museum }}</option>
		                    @endforeach
	                	</select>
	                	<span class="md-input-bar "></span>
	                </div>
                </div>                            
            </div>
            <div class="uk-width-medium-1-2">
            	<div class="md-card">
	                <div class="md-input-wrapper md-input-filled">
	                	<select name="historical_period_id" id="historical_period_id" class="md-input" data-uk-tooltip="{pos:'top'}">
		                    <option value="" disabled="" selected="" hidden="">دوره تاریخی</option>
		                    @foreach ($historicalPeriods as $i => $historicalPeriod)
		                    	<option value="{{ $i }}" {{ (old('historical_period_id') == $i || (isset($record) && !old('historical_period_id') && $record->historical_period_id == $i)) ? 'selected="selected"' : '' }}>{{ $historicalPeriod }}</option>
		                    @endforeach
	                	</select>
	                	<span class="md-input-bar "></span>
	                </div>
                </div>                            
            </div>
            <div class="uk-width-medium-1-2">
				<div class="uk-form-row md-card">
					<label>تعداد</label>
					<input type="text" name="count" value="{{ old('count') }}{{ (isset($record) && !old('count')) ? $record->count : '' }}" class="md-input" />
				</div>
			</div>
			<div class="uk-width-medium-1-2">
				<div class="uk-form-row md-card">
					<label>وزن (گرم)</label>
					<input type="text" name="weight" value="{{ old('weight') }}{{ (isset($record) && !old('weight')) ? $record->weight : '' }}" class="md-input" />
				</div>
			</div>
			<div class="uk-width-medium-1-2">
				<div class="uk-form-row md-card">
					<label>ابعاد (سانتی متر)</label>
					<input type="text" name="dimentions" value="{{ old('dimentions') }}{{ (isset($record) && !old('dimentions')) ? $record->dimentions : '' }}" class="md-input" />
				</div>
			</div>
			<div class="uk-width-medium-1-2">
				<div class="uk-form-row md-card">
					<label>ماده سازنده</label>
					<input type="text" name="material" value="{{ old('material') }}{{ (isset($record) && !old('material')) ? $record->material : '' }}" class="md-input" />
				</div>
			</div>
			<div class="uk-width-medium-1-2">
				<div class="uk-form-row md-card">
					<label>قدمت (سال)</label>
					<input type="text" name="age" value="{{ old('age') }}{{ (isset($record) && !old('age')) ? $record->age : '' }}" class="md-input" />
				</div>
			</div>
			<div class="uk-width-medium-1-2">
				<div class="uk-form-row md-card">
					<label>محل کشف</label>
					<input type="text" name="discovery_site" value="{{ old('discovery_site') }}{{ (isset($record) && !old('discovery_site')) ? $record->discovery_site : '' }}" class="md-input" />
				</div>
			</div>
			<div class="uk-width-medium-1-2">
                <div class="md-card">
                    <div class="md-card-content">
                    	<p class="heading_a uk-margin-bottom" style="min-height: 250px; text-align: center;">
	                        @if (isset($record) && $record->image1)
	                        		<img src="{{ URL::to($record->image1) }}" style="height: 250px;">
	                        @endif
                        </p>
                        <div class="uk-form-file md-btn md-btn-primary">
                            تصویر 1
                            <input name="image1" id="form-file" type="file">
                        </div>
                    </div>
                </div>
            </div>
            <div class="uk-width-medium-1-2">
                <div class="md-card">
                    <div class="md-card-content">
                    	<p class="heading_a uk-margin-bottom" style="min-height: 250px; text-align: center;">
	                        @if (isset($record) && $record->image2)
	                        	<img src="{{ URL::to($record->image2) }}" style="height: 250px;">
	                        @endif
                        </p>
                        <div class="uk-form-file md-btn md-btn-primary">
                            تصویر 2
                            <input name="image2" id="form-file" type="file">
                        </div>
                    </div>
                </div>
            </div>
            <div class="uk-width-medium-1-2">
                <div class="md-card">
                    <div class="md-card-content">
                    	<p class="heading_a uk-margin-bottom" style="min-height: 250px; text-align: center;">
	                        @if (isset($record) && $record->image3)
	                        	<img src="{{ URL::to($record->image3) }}" style="height: 250px;">
	                        @endif
                        </p>
                        <div class="uk-form-file md-btn md-btn-primary">
                            تصویر 3
                            <input name="image3" id="form-file" type="file">
                        </div>
                    </div>
                </div>
            </div>
            <div class="uk-width-medium-1-2">
                <div class="md-card">
                    <div class="md-card-content">
                    	<p class="heading_a uk-margin-bottom" style="min-height: 250px; text-align: center;">
	                        @if (isset($record) && $record->image4)
	                        	<img src="{{ URL::to($record->image4) }}" style="height: 250px;">
	                        @endif
                        </p>
                        <div class="uk-form-file md-btn md-btn-primary">
                            تصویر 4
                            <input name="image4" id="form-file" type="file">
                        </div>
                    </div>
                </div>
            </div>
		</div>
		<hr style="margin: 25px 0 15px 0">
		<div class="uk-grid">
			<div class="uk-width-1-7">
				<button type="submit" href="#" class="md-btn md-btn-success">ذخیره</button>
				<a href="{{ URL::to('items') }}"><button type="button" class="md-btn md-btn-default" style="margin-right: 10px;">انصراف</button></a>
			</div>
		</div>
	<form>
@endsection