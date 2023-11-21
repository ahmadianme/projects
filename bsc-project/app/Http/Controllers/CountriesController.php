<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;

use App\Http\Requests;
use App\Http\Controllers\Controller;

use App\Country as Model;

class CountriesController extends Controller
{
    public function __construct(){
        $this->middleware('auth');
    }
    
    public function index(){
        $query = new Model;

        if (isset($_POST['search_keyword'])){
            $searchKeyword = $_POST['search_keyword'];
            $query = $query->where('name' , 'like' , '%' . $searchKeyword . '%');
            $query = $query->orWhere('language' , 'like' , '%' . $searchKeyword . '%');
            $query = $query->orWhere('timezone' , 'like' , '%' . $searchKeyword . '%');
        }

        $records = $query->get();
        return view('countries.index' , ['records' => $records]);
    }

    public function create(){
        return view('countries.form');
    }

    public function store(Request $request){
        $this->validate($request, [
            'name' => 'required',
            'continent' => 'required',
            'language' => 'required',
            'timezone' => 'required',
        ]);

        Model::create([
            'user_id' => $request->user()->id,
            'name' => $request->name,
            'continent' => $request->continent,
            'language' => $request->language,
            'timezone' => $request->timezone,
        ]);

        return redirect('/countries');
    }

    public function show($id){
        //
    }

    public function edit($id){
        $record = Model::find($id);
        return view('countries.form' , ['record' => $record]);
    }

    public function update(Request $request, $id){
        $record = Model::find($id);

        $this->validate($request, [
            'name' => 'required',
            'continent' => 'required',
            'language' => 'required',
            'timezone' => 'required',
        ]);

        $record->name = $request->name;
        $record->continent = $request->continent;
        $record->language = $request->language;
        $record->timezone = $request->timezone;
        $record->save();

        return redirect('/countries');
    }

    public function destroy($id){
        $record = Model::find($id);
        $record->delete();

        return redirect('/countries');
    }
}
