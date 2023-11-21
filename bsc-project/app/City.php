<?php

namespace App;

use Illuminate\Database\Eloquent\Model;

class City extends Model
{
	protected $table = 'cities';

	protected $fillable = [
        'user_id',
		'country_id',
		'name',
	];

	public function user(){
        return $this->belongsTo('App\User');
    }
    
	public function country(){
        return $this->belongsTo('App\Country');
    }

    public function museums(){
        return $this->hasMany('App\Museum');
    }

    public static function getNameList(){
        $records = self::get();

        $list = [];
        foreach ($records as $record) {
            $list[$record->id] = $record->country->name . '/' . $record->name;
        }

        return $list;
    }
}
