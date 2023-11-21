<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;

use App\Http\Requests;
use App\Http\Controllers\Controller;

use App\Museum as Model;
use App\City;

class MuseumsController extends Controller
{
    public function __construct(){
        $this->middleware('auth');
    }
    
    public function index(){
        $query = new Model;

        if (isset($_POST['search_keyword'])){
            $searchKeyword = $_POST['search_keyword'];
            $query = $query->where('name' , 'like' , '%' . $searchKeyword . '%');
            $query = $query->orWhere('area' , 'like' , '%' . $searchKeyword . '%');
            $query = $query->orWhere('num_of_halls' , 'like' , '%' . $searchKeyword . '%');
            $query = $query->orWhere('phone' , 'like' , '%' . $searchKeyword . '%');
            $query = $query->orWhere('email' , 'like' , '%' . $searchKeyword . '%');
        }

        $records = $query->get();
        return view('museums.index' , ['records' => $records]);
    }

    public function create(){
        $cities = City::getNameList();
        return view('museums.form' , ['cities' => $cities]);
    }

    public function store(Request $request){
        $this->validate($request, [
            'name' => 'required',
            'city_id' => 'required',
            'area' => 'required',
            'num_of_halls' => 'required',
            'phone' => 'required',
            'email' => 'required',
            'address' => 'required',
        ]);

        Model::create([
            'user_id' => $request->user()->id,
            'name' => $request->name,
            'city_id' => $request->city_id,
            'area' => $request->area,
            'num_of_halls' => $request->num_of_halls,
            'phone' => $request->phone,
            'email' => $request->email,
            'address' => $request->address,
        ]);

        return redirect('/museums');
    }

    public function show($id){
        //
    }

    public function edit($id){
        $record = Model::find($id);
        $cities = City::getNameList();
        return view('museums.form' , ['record' => $record , 'cities' => $cities]);
    }

    public function update(Request $request, $id){
        $record = Model::find($id);

        $this->validate($request, [
            'name' => 'required',
            'city_id' => 'required',
            'area' => 'required',
            'num_of_halls' => 'required',
            'phone' => 'required',
            'email' => 'required',
            'address' => 'required',
        ]);

        $record->name = $request->name;
        $record->city_id = $request->city_id;
        $record->area = $request->area;
        $record->num_of_halls = $request->num_of_halls;
        $record->phone = $request->phone;
        $record->email = $request->email;
        $record->address = $request->address;
        $record->save();

        return redirect('/museums');
    }

    public function destroy($id){
        $record = Model::find($id);
        $record->delete();

        return redirect('/museums');
    }
}
