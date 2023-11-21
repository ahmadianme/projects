@extends('layouts.app')

@section('breadcrumbs')
	<li><span>خانه</span></li>
	<li><span>دوره های تاریخی</span></li>
@endsection

@section('content')
	<div class="uk-grid" style="margin-bottom: 25px;">
		<div class="uk-width-1-2">                    
			<a href="{{ URL::to('historicalperiods/new') }}" class="md-btn md-btn-primary md-btn-wave-light waves-effect waves-button waves-light" href="form.html">جدید</a>
		</div>

		<div class="uk-width-1-2">
			<form method="POST">
				{{ csrf_field() }}
				<div style="float: left;">
					<label>جستجو</label>
					<input type="text" name="search_keyword" value="{{ request('search_keyword') }}" class="md-input" />
				</div>
			</form>
		</div>
	</div>
	<hr>
	<div class="uk-overflow-container">
		@if (count($records))
			<table class="uk-table uk-table-nowrap table_check">
				<thead>
					<tr>
						<th class="uk-width-2-10">نام</th>
						<th class="uk-width-2-10 uk-text-center">سال شروع</th>
						<th class="uk-width-1-10 uk-text-center">سال پایان</th>
						<th class="uk-width-1-10 uk-text-center">کاربر</th>
						<th class="uk-width-2-10 uk-text-center">عملیات</th>
					</tr>
				</thead>
				<tbody>
					@foreach ($records as $record)
						<tr>
							<td>{{ $record->name }}</td>
							<td class="uk-text-center">{{ $record->begin_year }}</td>
							<td class="uk-text-center">{{ $record->end_year }}</td>
							<td class="uk-text-center">{{ isset($record->user) ? $record->user->name . ' ' . $record->user->lname : '-' }}</td>
							<td class="uk-text-center">
								<button onclick="window.location='{{ URL::to('historicalperiods/edit') }}/{{ $record->id }}';" class="md-btn md-btn-flat md-btn-wave waves-effect waves-button" style="padding: 0; min-width: 40px;"><i class="md-icon material-icons">&#xE254;</i></button>
								<form action="{{ URL::to('historicalperiods') }}/{{ $record->id }}" method="POST" id="deleteForm{{ $record->id }}" style="display: inline;">
						            {{ csrf_field() }}
						            {{ method_field('DELETE') }}

						            <button type="button" class="md-btn md-btn-flat md-btn-wave waves-effect waves-button" style="padding: 0; min-width: 40px;" onclick="UIkit.modal.confirm('آیا از حذف این مورد اطمینان دارید؟', function(){ $('#deleteForm{{ $record->id }}').submit(); });"><i class="md-icon material-icons">delete</i></button>
						        </form>
						        <button onclick="window.open('{{ URL::to('items/filterByHistoricalPeriod/' . $record->id) }}');" class="md-btn md-btn-flat md-btn-wave waves-effect waves-button" style="padding: 0; min-width: 40px;"><i class="md-icon material-icons">filter_vintage</i></button>
							</td>
						</tr>
					@endforeach
				</tbody>
			</table>
		@else
			هیچ موردی یافت نشد.
		@endif
	</div>
@endsection