<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;

use App\Http\Requests;
use App\Http\Controllers\Controller;

use App\HistoricalPeriod as Model;

class HistoricalPeriodsController extends Controller
{
    public function __construct(){
        $this->middleware('auth');
    }
    
    public function index(){
        $query = new Model;

        if (isset($_POST['search_keyword'])){
            $searchKeyword = $_POST['search_keyword'];
            $query = $query->where('name' , 'like' , '%' . $searchKeyword . '%');
            $query = $query->orWhere('begin_year' , 'like' , '%' . $searchKeyword . '%');
            $query = $query->orWhere('end_year' , 'like' , '%' . $searchKeyword . '%');
        }

        $records = $query->get();
        return view('historicalperiods.index' , ['records' => $records]);
    }

    public function create(){
        return view('historicalperiods.form');
    }

    public function store(Request $request){
        $this->validate($request, [
            'name' => 'required',
            'begin_year' => 'required',
            'end_year' => 'required',
        ]);

        Model::create([
            'user_id' => $request->user()->id,
            'name' => $request->name,
            'begin_year' => $request->begin_year,
            'end_year' => $request->end_year,
        ]);

        return redirect('/historicalperiods');
    }

    public function show($id){
        //
    }

    public function edit($id){
        $record = Model::find($id);
        return view('historicalperiods.form' , ['record' => $record]);
    }

    public function update(Request $request, $id){
        $record = Model::find($id);

        $this->validate($request, [
            'name' => 'required',
            'begin_year' => 'required',
            'end_year' => 'required',
        ]);

        $record->name = $request->name;
        $record->begin_year = $request->begin_year;
        $record->end_year = $request->end_year;
        $record->save();

        return redirect('/historicalperiods');
    }

    public function destroy($id){
        $record = Model::find($id);
        $record->delete();

        return redirect('/historicalperiods');
    }
}
