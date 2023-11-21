<?php

namespace App;

use Illuminate\Database\Eloquent\Model;

class Item extends Model
{
    protected $table = 'items';

	protected $fillable = [
        'user_id',
		'historical_period_id',
		'name',
		'count',
		'images',
	];

	public function user(){
        return $this->belongsTo('App\User');
    }

	public function historical_period(){
        return $this->belongsTo('App\HistoricalPeriod');
    }

    public function museum(){
        return $this->belongsTo('App\Museum');
    }
}
