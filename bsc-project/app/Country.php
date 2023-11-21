<?php

namespace App;

use Illuminate\Database\Eloquent\Model;

class Country extends Model
{
    protected $table = 'countries';

	protected $fillable = [
        'user_id',
		'name',
		'continent',
		'timezone',
		'language',
	];

	public function user(){
        return $this->belongsTo('App\User');
    }

	public function cities(){
        return $this->hasMany('App\City');
    }

    public static function getNameList(){
    	return self::lists('name' , 'id');
    }
}
