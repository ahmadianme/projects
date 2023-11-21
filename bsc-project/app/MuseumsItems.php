<?php

namespace App;

use Illuminate\Database\Eloquent\Model;

class MuseumsItems extends Model
{
    protected $table = 'museums_items';

	protected $fillable = [
        'user_id',
        'museum_id',
		'item_id',
	];

    public function user(){
        return $this->belongsTo('App\User');
    }

	public function museum(){
        return $this->belongsTo('App\Museum');
    }

    public function item(){
        return $this->belongsTo('App\Item');
    }
}
