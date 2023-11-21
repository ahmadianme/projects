<?php

use Illuminate\Database\Schema\Blueprint;
use Illuminate\Database\Migrations\Migration;

class CreateItemsTable extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        Schema::create('items', function (Blueprint $table) {
            $table->increments('id');
            $table->integer('user_id')->unsigned();
            $table->integer('museum_id')->unsigned();
            $table->integer('historical_period_id')->unsigned();
            $table->string('name' , 256);
            $table->integer('count')->default(1)->unsigned();
            $table->string('weight' , 16)->nullable();
            $table->string('dimentions' , 32)->nullable();
            $table->string('material' , 32)->nullable();
            $table->string('age' , 32);
            $table->string('discovery_site' , 128)->nullable();
            $table->text('image1')->nullable();
            $table->text('image2')->nullable();
            $table->text('image3')->nullable();
            $table->text('image4')->nullable();
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::drop('items');
    }
}
