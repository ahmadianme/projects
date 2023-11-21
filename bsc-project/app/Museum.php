<?php

namespace App;

use Illuminate\Database\Eloquent\Model;

class Museum extends Model
{
    protected $table = 'museums';

	protected $fillable = [
        'user_id',
		'city_id',
		'name',
		'area',
		'num_of_halls',
		'phone',
		'email',
		'address',
		'image1',
		'image2',
		'image3',
		'image4',
	];

	public function user(){
        return $this->belongsTo('App\User');
    }

	public function items(){
        return $this->hasMany('App\Item');
    }

    public function city(){
        return $this->belongsTo('App\City');
    }

    public static function getNameList(){
    	return self::lists('name' , 'id');
    }
}
