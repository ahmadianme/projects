<?php

namespace App;

use Illuminate\Database\Eloquent\Model;

class HistoricalPeriod extends Model
{
    protected $table = 'historical_periods';

	protected $fillable = [
		'user_id',
		'name',
		'begin_year',
		'end_year',
	];

	public function user(){
        return $this->belongsTo('App\User');
    }

	public function museums(){
        return $this->hasMany('App\Museum');
    }

    public static function getNameList(){
    	return self::lists('name' , 'id');
    }
}
