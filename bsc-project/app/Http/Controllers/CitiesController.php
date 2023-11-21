<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;

use App\Http\Requests;
use App\Http\Controllers\Controller;

use App\City as Model;
use App\Country;

class CitiesController extends Controller
{
    public function __construct(){
        $this->middleware('auth');
    }
    
    public function index(){
        $query = new Model;

        if (isset($_POST['search_keyword'])){
            $searchKeyword = $_POST['search_keyword'];
            $query = $query->where('name' , 'like' , '%' . $searchKeyword . '%');
        }

        $records = $query->get();
        return view('cities.index' , ['records' => $records]);
    }

    public function create(){
        $countries = Country::getNameList();
        return view('cities.form' , ['countries' => $countries]);
    }

    public function store(Request $request){
        $this->validate($request, [
            'name' => 'required',
            'country_id' => 'required',
        ]);

        Model::create([
            'user_id' => $request->user()->id,
            'name' => $request->name,
            'country_id' => $request->country_id,
        ]);

        return redirect('/cities');
    }

    public function show($id){
        //
    }

    public function edit($id){
        $record = Model::find($id);
        $countries = Country::getNameList();
        return view('cities.form' , ['record' => $record , 'countries' => $countries]);
    }

    public function update(Request $request, $id){
        $record = Model::find($id);

        $this->validate($request, [
            'name' => 'required',
            'country_id' => 'required',
        ]);

        $record->name = $request->name;
        $record->country_id = $request->country_id;
        $record->save();

        return redirect('/cities');
    }

    public function destroy($id){
        $record = Model::find($id);
        $record->delete();

        return redirect('/cities');
    }
}
