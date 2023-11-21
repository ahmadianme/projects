<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;

use App\Http\Requests;
use App\Http\Controllers\Controller;

use App\Http\Controllers\Auth\AuthController;
use App\User as Model;

class UsersController extends AuthController
{
    public function __construct(){
        $this->middleware('auth');
    }
    
    public function index(){
        $query = new Model;

        if (isset($_POST['search_keyword'])){
            $searchKeyword = $_POST['search_keyword'];
            $query = $query->where('name' , 'like' , '%' . $searchKeyword . '%');
            $query = $query->orWhere('lname' , 'like' , '%' . $searchKeyword . '%');
            $query = $query->orWhere('email' , 'like' , '%' . $searchKeyword . '%');
        }

        $records = $query->get();

        return view('users.index' , ['records' => $records]);
    }

    public function postRegister(Request $request){
        parent::postRegister($request);

        return redirect('/users');
    }

    public function edit($id){
        $record = Model::find($id);
        return view('auth.register' , ['record' => $record]);
    }

    public function update(Request $request, $id){
        $record = Model::find($id);

        $this->validate($request , [
            'name' => 'required|max:255',
            'lname' => 'required|max:255',
            'email' => 'required|email|max:255',
            'password' => 'confirmed|min:6',
        ]);

        $record->name = $request->name;
        $record->lname = $request->lname;
        $record->email = $request->email;
        if ($request->password)
            $record->password = bcrypt($request->password);
        $record->save();

        return redirect('/users');
    }

    public function destroy($id){
        $userId = \Auth::user()->id;
        if ($userId != $id){
            $record = Model::find($id);
            $record->delete();
        }

        return redirect('/users');
    }
}
