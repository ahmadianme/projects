<?php

use Illuminate\Database\Schema\Blueprint;
use Illuminate\Database\Migrations\Migration;

class CreateMuseumsTable extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        Schema::create('museums', function (Blueprint $table) {
            $table->increments('id');
            $table->integer('user_id')->unsigned();
            $table->integer('city_id')->unsigned();
            $table->string('name' , 256);
            $table->integer('area')->default(0)->unsigned();
            $table->tinyInteger('num_of_halls')->default(0)->unsigned();
            $table->string('phone' , 64)->nullable();
            $table->string('email' , 64)->nullable();
            $table->string('address' , 256)->nullable();
            $table->text('images')->nullable();
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
        Schema::drop('museums');
    }
}
