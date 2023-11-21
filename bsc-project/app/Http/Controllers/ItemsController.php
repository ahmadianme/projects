<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;

use App\Http\Requests;
use App\Http\Controllers\Controller;

use App\Item as Model;
use App\Museum;
use App\HistoricalPeriod;

class ItemsController extends Controller
{
    public function __construct(){
        $this->middleware('auth');
    }
    
    public function index($filters = []){
        $query = new Model;

        if (isset($filters['museumId'])){
            $query = $query->where('museum_id' , $filters['museumId']);
        }

        if (isset($filters['historicalPeriodId'])){
            $query = $query->where('historical_period_id' , $filters['historicalPeriodId']);
        }

        if (isset($_POST['search_keyword'])){
            $searchKeyword = $_POST['search_keyword'];
            $query = $query->where('name' , 'like' , '%' . $searchKeyword . '%');
        }

        $records = $query->get();

        return view('items.index' , ['records' => $records]);
    }

    public function indexFilterByMuseum($id){
        return $this->index(['museumId' => $id]);
    }

    public function indexFilterByHistoricalPeriod($id){
        return $this->index(['historicalPeriodId' => $id]);
    }

    public function create(){
        $museums = Museum::getNameList();
        $historicalPeriods = HistoricalPeriod::getNameList();
        return view('items.form' , ['museums' => $museums , 'historicalPeriods' => $historicalPeriods]);
    }

    public function store(Request $request){
        $this->validate($request, [
            'name' => 'required',
            'museum_id' => 'required',
            'historical_period_id' => 'required',
            'count' => 'required',
            'age' => 'required',
            'image1' => 'mimes:jpeg,bmp,png',
            'image2' => 'mimes:jpeg,bmp,png',
            'image3' => 'mimes:jpeg,bmp,png',
            'image4' => 'mimes:jpeg,bmp,png',
        ]);

        $record = new Model;
        $record->user_id = $request->user()->id;
        $record->name = $request->name;
        $record->museum_id = $request->museum_id;
        $record->historical_period_id = $request->historical_period_id;
        $record->count = $request->count;
        $record->weight = $request->weight;
        $record->dimentions = $request->dimentions;
        $record->material = $request->material;
        $record->age = $request->age;
        $record->discovery_site = $request->discovery_site;

        for ($i = 1 ; $i <= 4 ; $i++){
            if ($request->file('image' . $i)){
                $inputName = 'image' . $i;
                $imageName = $inputName . time() . '.' . $request->file($inputName)->getClientOriginalExtension();
                $request->file($inputName)->move(
                    base_path() . '/public/images/items/' , $imageName
                );
                $record->{$inputName} = '/images/items/' . $imageName;
            }
        }

        $record->save();

        // Model::create([
        //     'user_id' => $request->user()->id,
        //     'name' => $request->name,
        //     'museum_id' => $request->museum_id,
        //     'historical_period_id' => $request->historical_period_id,
        //     'count' => $request->count,
        // ]);

        return redirect('/items');
    }

    public function show($id){
        //
    }

    public function edit($id){
        $record = Model::find($id);
        $museums = Museum::getNameList();
        $historicalPeriods = HistoricalPeriod::getNameList();
        return view('items.form' , ['record' => $record , 'museums' => $museums , 'historicalPeriods' => $historicalPeriods]);
    }

    public function update(Request $request, $id){
        $record = Model::find($id);

        $this->validate($request, [
            'name' => 'required',
            'museum_id' => 'required',
            'historical_period_id' => 'required',
            'count' => 'required',
            'age' => 'required',
            'image1' => 'mimes:jpeg,bmp,png',
            'image2' => 'mimes:jpeg,bmp,png',
            'image3' => 'mimes:jpeg,bmp,png',
            'image4' => 'mimes:jpeg,bmp,png',
        ]);

        $record->name = $request->name;
        $record->museum_id = $request->museum_id;
        $record->historical_period_id = $request->historical_period_id;
        $record->count = $request->count;
        $record->weight = $request->weight;
        $record->dimentions = $request->dimentions;
        $record->material = $request->material;
        $record->age = $request->age;
        $record->discovery_site = $request->discovery_site;

        for ($i = 1 ; $i <= 4 ; $i++){
            if ($request->file('image' . $i)){
                $inputName = 'image' . $i;
                $imageName = $inputName . time() . '.' . $request->file($inputName)->getClientOriginalExtension();
                $request->file($inputName)->move(
                    base_path() . '/public/images/items/' , $imageName
                );
                $record->{$inputName} = '/images/items/' . $imageName;
            }
        }

        $record->save();

        return redirect('/items');
    }

    public function destroy($id){
        $record = Model::find($id);
        $record->delete();

        return redirect('/items');
    }
}
