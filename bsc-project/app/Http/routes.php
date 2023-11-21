<?php

/*
|--------------------------------------------------------------------------
| Application Routes
|--------------------------------------------------------------------------
|
| Here is where you can register all of the routes for an application.
| It's a breeze. Simply tell Laravel the URIs it should respond to
| and give it the controller to call when that URI is requested.
|
*/

Route::get('/', function () {
    return redirect('museums');
});

Route::get('home', function () {
    return redirect('museums');
});

//authemtication routes
Route::get('auth/login', 'Auth\AuthController@getLogin');
Route::post('auth/login', 'Auth\AuthController@postLogin');
Route::get('auth/logout', 'Auth\AuthController@getLogout');

//users controller routes
Route::any('users', 'UsersController@index');
Route::delete('users/{id}', 'UsersController@destroy');
Route::get('users/new', 'UsersController@getRegister');
Route::post('users/new', 'UsersController@postRegister');
Route::get('users/edit/{id}', 'UsersController@edit');
Route::post('users/edit/{id}', 'UsersController@update');

//museums controller routes
Route::any('museums', 'MuseumsController@index');
Route::delete('museums/{id}', 'MuseumsController@destroy');
Route::get('museums/new', 'MuseumsController@create');
Route::post('museums/new', 'MuseumsController@store');
Route::get('museums/edit/{id}', 'MuseumsController@edit');
Route::post('museums/edit/{id}', 'MuseumsController@update');

//items controller routes
Route::any('items', 'ItemsController@index');
Route::any('items/filterByMuseum/{museumId}', 'ItemsController@indexFilterByMuseum');
Route::any('items/filterByHistoricalPeriod/{historicalPeriodId}', 'ItemsController@indexFilterByHistoricalPeriod');
Route::delete('items/{id}', 'ItemsController@destroy');
Route::get('items/new', 'ItemsController@create');
Route::post('items/new', 'ItemsController@store');
Route::get('items/edit/{id}', 'ItemsController@edit');
Route::post('items/edit/{id}', 'ItemsController@update');

//countries controller routes
Route::any('countries', 'CountriesController@index');
Route::delete('countries/{id}', 'CountriesController@destroy');
Route::get('countries/new', 'CountriesController@create');
Route::post('countries/new', 'CountriesController@store');
Route::get('countries/edit/{id}', 'CountriesController@edit');
Route::post('countries/edit/{id}', 'CountriesController@update');

//cities controller routes
Route::any('cities', 'CitiesController@index');
Route::delete('cities/{id}', 'CitiesController@destroy');
Route::get('cities/new', 'CitiesController@create');
Route::post('cities/new', 'CitiesController@store');
Route::get('cities/edit/{id}', 'CitiesController@edit');
Route::post('cities/edit/{id}', 'CitiesController@update');

//historicalperiods controller routes
Route::any('historicalperiods', 'HistoricalPeriodsController@index');
Route::delete('historicalperiods/{id}', 'HistoricalPeriodsController@destroy');
Route::get('historicalperiods/new', 'HistoricalPeriodsController@create');
Route::post('historicalperiods/new', 'HistoricalPeriodsController@store');
Route::get('historicalperiods/edit/{id}', 'HistoricalPeriodsController@edit');
Route::post('historicalperiods/edit/{id}', 'HistoricalPeriodsController@update');